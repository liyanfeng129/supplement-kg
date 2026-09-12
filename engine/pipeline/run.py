"""The run orchestrator: records in, one reviewable row each out.

A central loader owns the queue and hands out the next record on request; N
workers each loop *take -> render -> call -> hand result to the appender*.
Records are independent, so there is no cross-worker coordination. One appender
thread flushes per row -- free at ~15s/record -- and workers never write
concurrently, so the output file cannot be corrupted by concurrency.

**Resume is input IDs minus IDs present in the output.** `interrupted` is not a
stored status: recording work-in-flight needs two writes and mutable rows, and
a crash kills the very code that would write it. Absence already means "not
done, for whatever reason", at a cost of one re-run record.

**The poison guard** counts hand-outs per ID in memory; after `MAX_ATTEMPTS` a
record is written as `status: poison` and skipped, so one pathological label
cannot stall a multi-day job across restarts.

**Progress is plain appended stdout lines on a timer** -- never ANSI
carriage-return redraw, which becomes thousands of garbage lines in a notebook
cell. Each line carries elapsed, done, remaining, rate over a *trailing* window
(a cumulative rate hides slowdowns), projected finish, the status breakdown,
and the running **grounding rate** -- the corruption alarm.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.pipeline import store
from engine.pipeline.config import Config, git_commit
from engine.pipeline.extract import LocalExtractor, check_gpu_offload, check_grammar
from engine.pipeline.validate import validate

# Hand-outs allowed per record before it is written off as poison.
MAX_ATTEMPTS = 3

PROGRESS_INTERVAL_S = 60
# Rate is averaged over this many recent completions, not over the whole run.
RATE_WINDOW = 50

# Minimum share of entries whose evidence must appear verbatim before a batch
# is allowed to proceed. A corrupted runtime emits fluent text that cannot
# reproduce exact substrings, so this is the cheap detector for it.
MIN_GROUNDING_RATE = 0.90
SMOKE_LABELS = 5

# The gold set, and the frozen passages it was built from. Smoke always reads
# its labels from here rather than from the batch under test.
DEFAULT_GOLD = "datasets/pilot/gold-100.jsonl"
DEFAULT_SMOKE_PASSAGES = "data/dsld/runs/20260908T125504Z/raw/pilot-1000.jsonl"


def load_records(path: str | Path, ids: list[int] | None = None, limit: int | None = None) -> list[dict]:
    """Read frozen `{dsld_id, name, passage}` records from a pilot-style JSONL.

    Reading passages rather than re-rendering labels keeps a run comparable to
    the pilot and the gold set, which were built from these exact passages.
    """
    out = []
    wanted = {int(i) for i in ids} if ids else None
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if wanted is not None and int(rec["dsld_id"]) not in wanted:
                continue
            out.append(rec)
            if limit and wanted is None and len(out) >= limit:
                break
    return out[:limit] if (limit and wanted is not None) else out


def _progress_line(
    started: float,
    done: int,
    total: int,
    recent: deque,
    statuses: Counter,
    grounded: int,
    entries: int,
) -> str:
    elapsed = time.monotonic() - started
    remaining = total - done
    rate = (len(recent) / (recent[-1] - recent[0])) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
    eta = "unknown"
    if rate > 0:
        eta = (datetime.now(timezone.utc) + timedelta(seconds=remaining / rate)).strftime("%Y-%m-%d %H:%M UTC")
    gr = f"{grounded / entries:.1%}" if entries else "n/a"
    breakdown = " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    return (
        f"[{timedelta(seconds=int(elapsed))}] done={done}/{total} remaining={remaining} "
        f"rate={rate * 60:.1f}/min eta={eta} grounding={gr} {breakdown}"
    )


def smoke(
    config: Config | None = None,
    records_path: str | Path | None = None,
    gold_path: str | Path = DEFAULT_GOLD,
    passages_path: str | Path = DEFAULT_SMOKE_PASSAGES,
    n: int = SMOKE_LABELS,
    min_grounding: float = MIN_GROUNDING_RATE,
    extractor: LocalExtractor | None = None,
) -> dict:
    """Put real gold labels through the real model, prompt and grammar.

    This is a live assertion, not a unit test, and it is part of the product:
    it exists to detect a runtime that loads normally and emits fluent garbage,
    which every mock would hide. A 2-token "ok" generation cannot tell a
    correct model from a corrupted one; reproducing exact substrings from a
    1,600-4,000 char passage can.

    Raises RuntimeError if the grounding rate falls below `min_grounding`.
    """
    config = config or Config()
    check_gpu_offload()
    check_grammar()

    gold_ids = [int(json.loads(l)["dsld_id"]) for l in Path(gold_path).read_text().splitlines() if l.strip()][:n]

    # Smoke passages come from the frozen pilot file, never from the batch's own
    # input. The gold IDs are a 100-label slice of the pilot, so a corpus run --
    # the batch this check matters most for -- need not contain them, and
    # sourcing them from `records_path` would make smoke refuse exactly there.
    records = load_records(passages_path, ids=gold_ids)
    if not records:
        raise RuntimeError(
            f"no smoke passages found for gold ids {gold_ids} in {passages_path}"
        )

    ex = extractor or LocalExtractor(config)
    rows = []
    for rec in records:
        raw, latency = ex.extract(rec["passage"])
        rows.append(
            validate(
                raw,
                rec["passage"],
                dsld_id=rec["dsld_id"],
                name=rec.get("name"),
                config_hash=config.config_hash,
                provenance=config.provenance,
                latency_ms=latency,
            )
        )

    entries = [e for r in rows for e in (*r["purpose"], *r["risk"])]
    n_grounded = sum(1 for e in entries if e["grounded"])
    rate = n_grounded / len(entries) if entries else 0.0

    result = {
        "n_labels": len(rows),
        "n_entries": len(entries),
        "n_grounded": n_grounded,
        "grounding_rate": rate,
        "statuses": dict(Counter(r["status"] for r in rows)),
        "median_latency_ms": sorted(r["latency_ms"] for r in rows)[len(rows) // 2],
        "config_hash": config.config_hash,
        "passed": bool(entries) and rate >= min_grounding,
    }
    if not result["passed"]:
        raise RuntimeError(
            f"SMOKE FAILED: grounding {rate:.1%} over {len(entries)} entries "
            f"(minimum {min_grounding:.0%}). The inference path may be "
            "corrupted -- this architecture has a known CUDA bug that loads "
            "normally and emits fluent garbage. Do not start a batch."
        )
    return result


def run(
    records_path: str | Path,
    config: Config | None = None,
    ids: list[int] | None = None,
    limit: int | None = None,
    n_workers: int = 1,
    resume: str | None = None,
    runs_root: str | Path = store.RUNS_ROOT,
    skip_smoke: bool = False,
    progress_interval: int = PROGRESS_INTERVAL_S,
    note: str = "",
) -> Path:
    """Extract over records, writing one row per attempt. Returns the run dir.

    A plain function: the CLI parses flags and calls this, and a notebook
    imports and calls the same thing -- one implementation, two callers.

    `resume` is an explicit run id, never automatic: appending to an existing
    run directory is always a deliberate act, and each resume's start time is
    recorded in the manifest so an appended-to directory still reads honestly.
    """
    config = config or Config()
    if note:
        config.note = note

    records = load_records(records_path, ids=ids, limit=limit)

    d = store.run_dir(resume, root=runs_root)
    started_at = datetime.now(timezone.utc).isoformat()

    already: set[int] = set()
    if resume:
        store.repair(d)
        already = store.done_ids(d)
        records = [r for r in records if int(r["dsld_id"]) not in already]

    # Smoke runs before every batch: a ggml upgrade that silently corrupts the
    # DeltaNet path must not be allowed to poison a multi-day run.
    extractor = LocalExtractor(config)
    smoke_result = None
    if not skip_smoke:
        print("smoke: putting gold labels through the real model...", flush=True)
        smoke_result = smoke(config, extractor=extractor)
        print(
            f"smoke: passed, grounding {smoke_result['grounding_rate']:.1%} "
            f"over {smoke_result['n_entries']} entries "
            f"(median {smoke_result['median_latency_ms']}ms/label)",
            flush=True,
        )

    manifest = {
        "task": "dsld_purpose_risk",
        "run_id": d.name,
        "started_at": started_at,
        # Each resume appends, so the record stays honest about restarts.
        "starts": [*(store.read_manifest(d).get("starts") or []), started_at],
        # In the manifest, not the hash: two identical runs a week apart pool.
        "git_commit": git_commit(),
        "records_path": str(records_path),
        "n_input": len(records) + len(already),
        "n_already_done": len(already),
        "n_to_do": len(records),
        "n_workers": n_workers,
        "smoke": smoke_result,
        **config.resolved(),
    }
    store.write_manifest(d, manifest)

    if not records:
        print(f"nothing to do; {len(already)} rows already in {d}", flush=True)
        return d

    pending: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    lock = threading.Lock()

    # Attempts already spent on each ID, counted from the run's own rows rather
    # than held only in memory. A record that kills the process leaves no row at
    # all, so absence cannot distinguish "never tried" from "crashed the
    # interpreter"; what it can count is prior failed attempts that did get
    # written. Combined with `resume`, this is what stops one pathological label
    # from being retried forever across restarts.
    prior_attempts: Counter = Counter(
        int(r["dsld_id"]) for r in store.read_rows(d) if r.get("status") == "error"
    )
    attempts: Counter = Counter()

    for rec in records:
        pending.put(rec)

    def take():
        """Hand out the next record, writing off anything tried too often."""
        while True:
            try:
                rec = pending.get_nowait()
            except queue.Empty:
                return None
            did = int(rec["dsld_id"])
            with lock:
                attempts[did] += 1
                n = attempts[did] + prior_attempts[did]
            if n > MAX_ATTEMPTS:
                results.put(
                    validate(
                        "",
                        rec["passage"],
                        dsld_id=rec["dsld_id"],
                        name=rec.get("name"),
                        config_hash=config.config_hash,
                        provenance=config.provenance,
                        status="poison",
                        error=f"failed {n - 1} times already; not retried again",
                    )
                )
                continue
            return rec

    def requeue(rec: dict) -> bool:
        """Put a failed record back, if it has attempts left.

        Without this the poison guard is unreachable: a record handed out once
        and errored is never offered again, so the count never grows. Returning
        False means the caller should write the error row itself.
        """
        did = int(rec["dsld_id"])
        with lock:
            spent = attempts[did] + prior_attempts[did]
        # Hand it back while the budget allows another *attempt*. Once it does
        # not, `take` is still the place that writes the record off, so the
        # record goes back one final time and comes out of `take` as `poison`
        # rather than as one more `error`.
        if spent > MAX_ATTEMPTS:
            return False
        pending.put(rec)
        return True

    def worker():
        while True:
            rec = take()
            if rec is None:
                return
            try:
                raw, latency = extractor.extract(rec["passage"])
                row = validate(
                    raw,
                    rec["passage"],
                    dsld_id=rec["dsld_id"],
                    name=rec.get("name"),
                    config_hash=config.config_hash,
                    provenance=config.provenance,
                    latency_ms=latency,
                )
            except Exception as exc:  # noqa: BLE001 - one bad label must not sink a run
                # No inline retry of the *call*: the failure rate must stay
                # visible. This re-queues the record so the poison guard can
                # count it, and the attempt is written as its own `error` row
                # either way once the budget is spent.
                if requeue(rec):
                    continue
                row = validate(
                    "",
                    rec["passage"],
                    dsld_id=rec["dsld_id"],
                    name=rec.get("name"),
                    config_hash=config.config_hash,
                    provenance=config.provenance,
                    status="error",
                    error=repr(exc),
                )
            results.put(row)

    total = len(records)
    statuses: Counter = Counter()
    recent: deque = deque(maxlen=RATE_WINDOW)
    grounded = entries = done = 0
    run_started = time.monotonic()
    last_report = run_started

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, n_workers))]
    for w in workers:
        w.start()

    # One appender: workers never write concurrently, so no row can be torn.
    with store.rows_path(d).open("a") as fh:
        while done < total:
            try:
                row = results.get(timeout=1.0)
            except queue.Empty:
                if not any(w.is_alive() for w in workers) and results.empty():
                    break
                continue

            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

            done += 1
            statuses[row["status"]] += 1
            recent.append(time.monotonic())
            for e in (*row["purpose"], *row["risk"]):
                entries += 1
                grounded += bool(e["grounded"])

            now = time.monotonic()
            if now - last_report >= progress_interval or done == total:
                print(
                    _progress_line(run_started, done, total, recent, statuses, grounded, entries),
                    flush=True,
                )
                last_report = now

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["n_written"] = done
    manifest["statuses"] = dict(statuses)
    manifest["grounding_rate"] = (grounded / entries) if entries else None
    store.write_manifest(d, manifest)

    print(f"run complete: {d}", flush=True)
    return d


def retry_ids(d: str | Path) -> list[int]:
    """IDs in a run that did not come back `ok` or `empty`.

    The input to a retry pass: `error`, `unparseable` and `poison` rows, which
    are the machinery failing rather than the model correctly finding nothing.
    """
    return sorted(
        int(r["dsld_id"])
        for r in store.read_rows(d)
        if r.get("status") not in {"ok", "empty"}
    )


def retry(
    d: str | Path,
    records_path: str | Path,
    config: Config | None = None,
    **kwargs,
) -> Path | None:
    """Re-run a run's non-`ok` IDs into **its own new run directory**.

    Deliberately not an in-place fix and not an inline retry. A retried record
    is attributable rather than blended into the original run, and the original
    run's failure rate stays readable -- it is the primary early signal that a
    prompt or grammar is wrong. Returns None when there is nothing to retry.
    """
    ids = retry_ids(d)
    if not ids:
        print(f"nothing to retry in {d}", flush=True)
        return None

    note = kwargs.pop("note", "") or f"retry of {Path(d).name} ({len(ids)} non-ok rows)"
    print(f"retrying {len(ids)} non-ok records from {Path(d).name}", flush=True)
    return run(records_path, config=config, ids=ids, note=note, **kwargs)


def stats(d: str | Path) -> dict:
    """Summarise a run: status breakdown, grounding rate, latency."""
    rows = store.read_rows(d)
    entries = [e for r in rows for e in (*r.get("purpose", []), *r.get("risk", []))]
    lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    n_grounded = sum(1 for e in entries if e["grounded"])
    return {
        "run_dir": str(d),
        "n_rows": len(rows),
        "statuses": dict(Counter(r.get("status") for r in rows)),
        "n_entries": len(entries),
        "grounding_rate": (n_grounded / len(entries)) if entries else None,
        "n_ungrounded_entries": len(entries) - n_grounded,
        "n_with_purpose": sum(1 for r in rows if r.get("purpose")),
        "n_with_risk": sum(1 for r in rows if r.get("risk")),
        "median_latency_ms": lat[len(lat) // 2] if lat else None,
        "config_hash": store.read_manifest(d).get("config_hash"),
    }
