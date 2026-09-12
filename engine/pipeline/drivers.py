"""The `api` and `agent` drivers. The `local` driver is `run.run()`.

The enforced abstraction is **the output row plus `validate()`**, not a
`Backend` class. A uniform interface was rejected deliberately: subagents can
only be spawned from inside an agent's turn -- there is no API a script can
call -- so an `AgentBackend` would raise `NotImplementedError` when driven from
a loop, and an interface that lies is worse than two entry points.

So each driver owns its own loop and writes conforming rows. Prompt, schema,
grammar, grounding check, confidence cap, store and scorer are shared.
`provenance` is a column, never a branch. `latency_ms` is null for the agent
driver, whose wall-clock is not comparable to inference latency.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from engine.pipeline import store
from engine.pipeline.config import Config, git_commit
from engine.pipeline.run import load_records
from engine.pipeline.validate import validate


def _manifest(d: Path, config: Config, records_path, n_input: int, provenance: str) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    return {
        "task": "dsld_purpose_risk",
        "run_id": d.name,
        "started_at": started,
        "starts": [*(store.read_manifest(d).get("starts") or []), started],
        "git_commit": git_commit(),
        "records_path": str(records_path),
        "n_input": n_input,
        **{**config.resolved(), "provenance": provenance},
    }


def run_api(
    records_path: str | Path,
    config: Config | None = None,
    ids: list[int] | None = None,
    limit: int | None = None,
    model: str | None = None,
    runs_root: str | Path = store.RUNS_ROOT,
    resume: str | None = None,
    note: str = "",
) -> Path:
    """The Anthropic driver. Needs a key this machine does not have.

    Kept working rather than deleted: the pipeline must still run on a machine
    that does have one, and the gold set was built through this path.
    """
    from engine.services.extraction.label_purpose_risk import MODEL, SCHEMA, _client

    config = config or Config(provenance="api")
    config.provenance = "api"
    model = model or MODEL
    config.model_id = model
    if note:
        config.note = note

    client = _client()  # raises with a clear message when the key is absent
    records = load_records(records_path, ids=ids, limit=limit)

    d = store.run_dir(resume, root=runs_root)
    if resume:
        store.repair(d)
        done = store.done_ids(d)
        records = [r for r in records if int(r["dsld_id"]) not in done]

    store.write_manifest(d, _manifest(d, config, records_path, len(records), "api"))

    with store.rows_path(d).open("a") as fh:
        for rec in records:
            started = time.monotonic()
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=config.decode.max_tokens,
                    system=config.prompt,
                    output_config={
                        "effort": "low",
                        "format": {"type": "json_schema", "schema": SCHEMA},
                    },
                    messages=[{"role": "user", "content": rec["passage"]}],
                )
                raw = next((b.text for b in response.content if b.type == "text"), "")
                row = validate(
                    raw,
                    rec["passage"],
                    dsld_id=rec["dsld_id"],
                    name=rec.get("name"),
                    config_hash=config.config_hash,
                    provenance="api",
                    latency_ms=int((time.monotonic() - started) * 1000),
                )
            except Exception as exc:  # noqa: BLE001 - one bad label must not sink a run
                row = validate(
                    "",
                    rec["passage"],
                    dsld_id=rec["dsld_id"],
                    name=rec.get("name"),
                    config_hash=config.config_hash,
                    provenance="api",
                    status="error",
                    error=repr(exc),
                )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    return d


def start_agent_run(
    records_path: str | Path,
    config: Config | None = None,
    ids: list[int] | None = None,
    limit: int | None = None,
    runs_root: str | Path = store.RUNS_ROOT,
    resume: str | None = None,
    note: str = "",
) -> tuple[Path, list[dict]]:
    """Open a run directory for an in-session agent pass.

    Returns `(run_dir, records)`. The agent fans records out to subagents in
    its own turn -- there is no loop here to call, which is exactly why this is
    not a `Backend`. Each result comes back through `append_agent_row()`.
    """
    config = config or Config(provenance="agent")
    config.provenance = "agent"
    if note:
        config.note = note

    records = load_records(records_path, ids=ids, limit=limit)
    d = store.run_dir(resume, root=runs_root)
    if resume:
        store.repair(d)
        done = store.done_ids(d)
        records = [r for r in records if int(r["dsld_id"]) not in done]

    store.write_manifest(d, _manifest(d, config, records_path, len(records), "agent"))
    return d, records


def append_agent_row(
    run_dir: str | Path,
    raw: dict | str,
    passage: str,
    dsld_id: int,
    config: Config,
    name: str | None = None,
) -> dict:
    """Validate one agent-produced result and append it to the run.

    `latency_ms` stays null: a subagent's wall-clock includes scheduling and
    reading, and reporting it beside inference latency would invite a
    comparison that does not hold.
    """
    row = validate(
        raw,
        passage,
        dsld_id=dsld_id,
        name=name,
        config_hash=config.config_hash,
        provenance="agent",
        latency_ms=None,
    )
    with store.rows_path(run_dir).open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
