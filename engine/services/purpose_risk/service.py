"""Purpose/risk over DSLD labels: loader -> controller -> store, and nothing else.

**The service only coordinates.** It loads the prompt, writes or checks the
run's manifest, and owns the one loop: each label becomes a `Prompt` as the
controller pulls it, each outcome becomes a row, and each row goes to the
sink. It does no parsing and no checking (the controller runs the prompt's
own), no retrying, no progress or per-label logging (the controller's), and
knows nothing of how the model is hosted. Boilerplate filtering, grounding and
scoring will be added here later; none of it is in this build.

**It declares its own collaborators.** It is for purpose/risk only, so it
loads the `purpose_risk` prompt itself and creates the controller, which
creates the model. `controller` is the seam tests use to pass one built on a
fake model.

**Every label gets a row**, failed ones included, so a resume skips it:
`ok`, `failed_checks`, `too_long` or `error`. A failed row keeps the raw text
and empty `purpose` / `risk`. Failures never stop the run; once more than
`[notify].fail_rate` of this session's labels have failed, after at least
`[notify].fail_min_labels`, a person is told -- once. Only `ModelUnavailable`
stops a run: the finished rows are flushed, a person is told, and it is
re-raised.

**A run is created whole, then resumed from its own files.** `run=None`
starts a new one: the label set is loaded by name, the ids are chosen from it
-- `ids`, or the whole set, then `order` and `limit` -- and the store creates
the run, named after the set, with those ids and a manifest holding the set's
name and `passages_sha`, the selection arguments and the `config_hash`.
`run=<path>` resumes it: the ids and the set come from the run, never from the
call. A `config_hash` or `passages_sha` that differs from the manifest's is
refused -- the answers would no longer be comparable. The fix is a new run.

A label is done when its id has a row. Resume is the run's ids minus those.

`config_hash` covers what changes an answer: the prompt (system text, stop
sequences, output budget, check names), the model, its sampling settings and
the attempt budget. Not `in_flight`, `n_parallel` or the flush settings: they
change speed, not answers.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

from engine import config
from engine.controllers.qwen_controller import ERROR, TOO_LONG, QwenController
from engine.models.base_model import ModelUnavailable, Prompt
from engine.notify import notify
from engine.repositories.dsld import loader, store
from engine.repositories.prompts import loader as prompts

PROMPT_ID = "purpose_risk"

# Used when `configs/engine.toml` has no `[notify]` value.
_FALLBACK_FAIL_RATE = 0.05
_FALLBACK_FAIL_MIN_LABELS = 200

# The model settings that change an answer, as `configs/qwen.toml` [sampling] names them.
_SAMPLING = ("seed", "temperature", "top_p", "top_k", "max_tokens", "enable_thinking")


class PurposeRiskService:
    """Runs the purpose/risk prompt over DSLD labels into a run directory."""

    def __init__(self, controller: QwenController | None = None):
        """Creates the controller -- and through it the model -- unless given one."""
        self.controller = QwenController() if controller is None else controller

    def __enter__(self) -> "PurposeRiskService":
        return self

    def __exit__(self, *exc) -> None:
        self.controller.close()

    def run(
        self,
        label_set: str | None = None,
        ids: list[int] | None = None,
        limit: int | None = None,
        order: str | None = None,
        seed: int | None = None,
        run: str | Path | None = None,
    ) -> None:
        """Run every label of the run that has no row yet.

        A new run: `label_set` is a set's name. `ids` picks labels
        from it -- every id must be in the set -- or, left out, the whole set
        is taken, in its file order. `order="sorted"` sorts by id;
        `order="random"` shuffles with `seed`, which is then required.
        `limit` keeps the first `limit` after ordering.

        `run=<path>` resumes that run; `label_set` and the selection arguments
        must then be left out.

        Returns nothing. The rows are in the run directory; one line is
        printed at the end, and `notify` is told the counts.
        """
        spec = prompts.load(PROMPT_ID)

        # 1. What changes an answer, hashed. The model is reached through the
        # controller; a stand-in without Qwen's settings hashes by its name.
        model = self.controller.model
        answer_settings = {
            "prompt": {
                "id": PROMPT_ID,
                "system": spec.system,
                "stop": list(spec.stop),
                "max_tokens": spec.max_tokens,
                "checks": [c.name for c in spec.checks],
            },
            "model": getattr(model, "model_path", type(model).__name__),
            "sampling": {k: getattr(model, k, None) for k in _SAMPLING},
            "max_attempts": self.controller.max_attempts,
        }
        config_hash = hashlib.sha256(
            json.dumps(answer_settings, sort_keys=True).encode()
        ).hexdigest()[:16]

        # 2. The run: created with its ids, or reopened and checked.
        selection = {
            "ids": None if ids is None else [int(i) for i in ids],
            "limit": limit,
            "order": order,
            "seed": seed,
        }
        if run is None:
            # A new run: its ids are chosen here, once, and stored with it.
            if label_set is None:
                raise ValueError("a new run needs label_set=<name>")
            passages_sha, labels = loader.load_flattened(label_set)

            # Which labels: the ids named, or the whole set in its file order.
            # Every named id must be in the set, and named once.
            chosen = list(labels) if selection["ids"] is None else selection["ids"]
            if len(set(chosen)) != len(chosen):
                raise ValueError("ids holds an id more than once")
            missing = [i for i in chosen if i not in labels]
            if missing:
                shown = ", ".join(str(i) for i in missing[:5])
                more = f" (+{len(missing) - 5})" if len(missing) > 5 else ""
                raise ValueError(
                    f"{len(missing)} of {len(chosen)} id(s) are not in label set "
                    f"{label_set!r}: {shown}{more}"
                )

            # In what order.
            match order, seed:
                case None, None:
                    pass  # as named, or the set's file order
                case "sorted", None:
                    chosen = sorted(chosen)
                case "random", None:
                    raise ValueError(
                        "order='random' needs seed=..., or the draw cannot be redrawn"
                    )
                case "random", _:
                    # Shuffled from sorted, so the draw depends on the seed and
                    # the set of ids -- not on the order they were named in.
                    chosen = sorted(chosen)
                    random.Random(seed).shuffle(chosen)
                case _, None:
                    raise ValueError(f"order must be 'sorted' or 'random', not {order!r}")
                case _:
                    raise ValueError("seed only means something with order='random'")

            # How many: the first `limit` of them.
            if limit is not None:
                chosen = chosen[:limit]

            run_path = store.new_run(
                label_set,
                chosen,
                {
                    "prompt": PROMPT_ID,
                    "label_set": label_set,
                    "passages_sha": passages_sha,
                    "selection": selection,
                    "config_hash": config_hash,
                },
            )
        else:#run is not none
            # A resume: everything comes from the run, and is refused if it no
            # longer matches what the run started with.
            run_path = Path(run)
            manifest = loader.read_manifest(run_path)
            if not manifest:
                raise FileNotFoundError(
                    f"no manifest at {store.manifest_path(run_path)}: not a run to resume. "
                    "Pass run=None to start a new one."
                )

            # The selection is the run's; passing one again would be ignored.
            if label_set is not None or any(v is not None for v in selection.values()):
                raise ValueError(
                    f"resuming {run_path}: its label set and ids come from the run; "
                    "leave label_set, ids, limit, order and seed out."
                )

            # Answers made another way would not be comparable.
            if manifest.get("config_hash") != config_hash:
                raise ValueError(
                    f"refusing to resume {run_path}: its config_hash is "
                    f"{manifest.get('config_hash')!r} and this configuration's is "
                    f"{config_hash!r} -- the prompt, model, sampling or attempt budget "
                    "changed, so its rows would not be comparable. Start a new run."
                )

            # Nor would answers to passages that have changed under the set's name.
            passages_sha, labels = loader.load_flattened(manifest["label_set"])
            if passages_sha != manifest.get("passages_sha"):
                raise ValueError(
                    f"refusing to resume {run_path}: label set {manifest['label_set']!r} "
                    f"now has passages_sha {passages_sha}, and the run started on "
                    f"{manifest.get('passages_sha')}. Start a new run."
                )

        # 3. What is left: the run's ids without a row, each label built into
        # a prompt only as the controller pulls it.
        run_ids = loader.read_ids(run_path)
        finished = loader.done_ids(run_path)
        todo = [i for i in run_ids if i not in finished]
        n_already_done = len(run_ids) - len(todo)
        work = (
            (
                i,
                Prompt(
                    system=spec.system,
                    user=labels[i].passage,
                    stop=spec.stop,
                    max_tokens=spec.max_tokens,
                ),
            )
            for i in todo
        )

        table = config.read_toml(config.ENGINE_CONFIG_PATH).get("notify")
        table = table if isinstance(table, dict) else {}
        fail_rate, fail_min_labels = table.get("fail_rate"), table.get("fail_min_labels")
        if not isinstance(fail_rate, (int, float)) or isinstance(fail_rate, bool):
            fail_rate = _FALLBACK_FAIL_RATE
        if not isinstance(fail_min_labels, int) or isinstance(fail_min_labels, bool):
            fail_min_labels = _FALLBACK_FAIL_MIN_LABELS

        # 4. The loop: outcome -> row -> sink. Cheap, because it runs on the
        # thread that keeps the model fed.
        counts: collections.Counter[str] = collections.Counter()
        warned = False
        with store.Sink(run_path) as sink:
            try:
                for dsld_id, outcome in self.controller.stream(
                    work,
                    spec.checks,
                    spec.parse,
                    total=len(todo),
                    n_already_done=n_already_done,
                ):
                    if outcome.ok:
                        status = "ok"
                    elif outcome.failed == [TOO_LONG]:
                        status = "too_long"
                    elif outcome.failed == [ERROR]:
                        status = "error"
                    else:
                        status = "failed_checks"
                    parsed = outcome.parsed or {}
                    generation = outcome.generation
                    sink.add(
                        {
                            "dsld_id": dsld_id,
                            "status": status,
                            "purpose": parsed.get("purpose", []),
                            "risk": parsed.get("risk", []),
                            "text": generation.text,
                            "attempts": outcome.attempts,
                            "failed": outcome.failed,
                            "prompt_tokens": generation.usage.prompt_tokens,
                            "completion_tokens": generation.usage.completion_tokens,
                            "latency_ms": generation.latency_ms,
                            "config_hash": config_hash,
                        }
                    )

                    counts[status] += 1
                    done = sum(counts.values())
                    n_failed = done - counts["ok"]
                    if not warned and done >= fail_min_labels and n_failed / done > fail_rate:
                        warned = True
                        notify(
                            f"failure rate over {fail_rate:.0%}",
                            f"{run_path}: {n_failed} of {done} labels this session have "
                            f"failed ({dict(counts)}). The run goes on; this is the only warning.",
                        )
            except ModelUnavailable as exc:
                sink.flush()
                notify(
                    "model unavailable",
                    f"{run_path}: the run stopped after {sum(counts.values())} labels this "
                    f"session ({dict(counts)}); every finished row is on disk. Resume with "
                    f"run={str(run_path)!r} once the model can run.\n\n{exc}",
                )
                raise

        # 5. The end: one notice, one line.
        summary = ", ".join(
            f"{status} {counts[status]}" for status in ("ok", "failed_checks", "too_long", "error")
        )
        line = (
            f"{run_path}: {sum(counts.values())} labels this session ({summary}); "
            f"{n_already_done} done before"
        )
        notify("run finished", line)
        print(line)
