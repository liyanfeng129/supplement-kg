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

**Resume is a run directory.** `run=None` starts a new timestamped one and
writes the manifest -- the selection arguments and the `config_hash`.
`run=<path>` resumes it: the selection is read from the manifest, never passed
again, and a `config_hash` that differs from the manifest's is refused -- the
answers would no longer be comparable. The fix is a new run directory.

`config_hash` covers what changes an answer: the prompt (system text, stop
sequences, output budget, check names), the model, its sampling settings and
the attempt budget. Not `in_flight`, `n_parallel` or the flush settings: they
change speed, not answers.
"""

from __future__ import annotations

import collections
import hashlib
import json
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
        run: str | Path | None = None,
        path: str | Path | None = None,
        ids: list[int] | None = None,
        limit: int | None = None,
        order: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Run every label the selection names that the run does not hold yet.

        `run=None` starts a new run with this selection -- `path`, `ids`,
        `limit`, `order`, `seed`, meaning what they mean to `loader.pending`.
        `run=<path>` resumes that run with the selection in its manifest; the
        selection arguments must then be left out. `limit` counts the run's
        labels, not a session's: a resumed run of 100 with 40 done does 60.

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

        # 2. The manifest: written for a new run, checked for a resumed one.
        selection = {
            "path": None if path is None else str(path),
            "ids": None if ids is None else [int(i) for i in ids],
            "limit": limit,
            "order": order,
            "seed": seed,
        }
        if run is None:
            run_path = store.run_dir(root=store.RUNS_ROOT)
            store.write_manifest(
                run_path,
                {"prompt": PROMPT_ID, "selection": selection, "config_hash": config_hash},
            )
        else:
            run_path = Path(run)
            manifest = store.read_manifest(run_path)
            if not manifest:
                raise FileNotFoundError(
                    f"no manifest at {store.manifest_path(run_path)}: not a run to resume. "
                    "Pass run=None to start a new one."
                )
            if any(v is not None for v in selection.values()):
                raise ValueError(
                    f"resuming {run_path}: the selection comes from its manifest; "
                    "leave path, ids, limit, order and seed out."
                )
            if manifest.get("config_hash") != config_hash:
                raise ValueError(
                    f"refusing to resume {run_path}: its config_hash is "
                    f"{manifest.get('config_hash')!r} and this configuration's is "
                    f"{config_hash!r} -- the prompt, model, sampling or attempt budget "
                    "changed, so its rows would not be comparable. Start a new run."
                )
            selection = manifest["selection"]

        # 3. What is left, read lazily. `limit` is the run's, so what earlier
        # sessions wrote counts against it.
        run_limit = selection["limit"]
        if run_limit is not None:
            run_limit = max(0, run_limit - len(store.done_ids(run_path)))
        pending = loader.pending(
            run_path,
            selection["path"],
            ids=selection["ids"],
            limit=run_limit,
            order=selection["order"],
            seed=selection["seed"],
        )
        work = (
            (
                label.dsld_id,
                Prompt(
                    system=spec.system,
                    user=label.passage,
                    stop=spec.stop,
                    max_tokens=spec.max_tokens,
                ),
            )
            for label in pending
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
                    total=pending.total,
                    n_already_done=pending.n_already_done,
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
            f"{pending.n_already_done} done before"
        )
        notify("run finished", line)
        print(line)
