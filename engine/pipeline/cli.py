"""Argparse only, no logic.

Every subcommand parses flags and calls the same function a notebook imports,
so behaviour cannot diverge between the two callers.

    python -m engine.pipeline.cli smoke  --records data/.../pilot-1000.jsonl
    python -m engine.pipeline.cli run    --records ... --limit 20 --note "..."
    python -m engine.pipeline.cli retry  --run data/dsld/runs/<id> --records ...
    python -m engine.pipeline.cli score  --run data/dsld/runs/<id>
    python -m engine.pipeline.cli stats  --run data/dsld/runs/<id>
    python -m engine.pipeline.cli compare --run-a <id> --run-b <id> --records ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.pipeline import run as run_mod
from engine.pipeline import store
from engine.pipeline.compare import compare
from engine.pipeline.config import Config, Decode
from engine.pipeline.score import score

DEFAULT_RECORDS = "data/dsld/runs/20260908T125504Z/raw/pilot-1000.jsonl"
DEFAULT_GOLD = "datasets/pilot/gold-100.jsonl"


def _config(args) -> Config:
    decode = Decode()
    if getattr(args, "max_tokens", None):
        decode.max_tokens = args.max_tokens
    if getattr(args, "n_ctx", None):
        decode.n_ctx = args.n_ctx
    if getattr(args, "temperature", None) is not None:
        decode.temperature = args.temperature
    cfg = Config(decode=decode, note=getattr(args, "note", "") or "")
    if getattr(args, "prompt", None):
        cfg.prompt_path = args.prompt
    if getattr(args, "model", None):
        cfg.model_id = args.model
    return cfg


def _gold(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pipeline", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--records", default=DEFAULT_RECORDS)
        sp.add_argument("--prompt")
        sp.add_argument("--model")
        sp.add_argument("--max-tokens", type=int, dest="max_tokens")
        sp.add_argument("--n-ctx", type=int, dest="n_ctx")
        sp.add_argument("--temperature", type=float)

    sp = sub.add_parser("smoke", help="real gold labels through the real model")
    common(sp)
    sp.add_argument("--gold", default=DEFAULT_GOLD)
    sp.add_argument("-n", type=int, default=run_mod.SMOKE_LABELS)
    sp.add_argument("--min-grounding", type=float, default=run_mod.MIN_GROUNDING_RATE)

    sp = sub.add_parser("run", help="extract over records into a new run dir")
    common(sp)
    sp.add_argument("--ids", help="comma-separated DSLD IDs")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--resume", help="existing run id to append to (never automatic)")
    sp.add_argument("--note", default="", help="what you were trying to achieve")
    sp.add_argument("--skip-smoke", action="store_true")

    sp = sub.add_parser("retry", help="re-run a run's non-ok IDs into a new run dir")
    common(sp)
    sp.add_argument("--run", required=True, help="run id whose non-ok rows to retry")
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--note", default="")
    sp.add_argument("--skip-smoke", action="store_true")

    sp = sub.add_parser("score", help="score a run against the gold set")
    sp.add_argument("--run", required=True)
    sp.add_argument("--gold", default=DEFAULT_GOLD)

    sp = sub.add_parser("stats", help="status breakdown and grounding rate")
    sp.add_argument("--run", required=True)

    sp = sub.add_parser("compare", help="prompt diff above metric delta")
    sp.add_argument("--run-a", required=True, dest="run_a")
    sp.add_argument("--run-b", required=True, dest="run_b")
    sp.add_argument("--gold", default=DEFAULT_GOLD)

    args = p.parse_args(argv)

    if args.cmd == "smoke":
        print(json.dumps(
            run_mod.smoke(
                _config(args),
                records_path=args.records,
                gold_path=args.gold,
                n=args.n,
                min_grounding=args.min_grounding,
            ),
            indent=2,
        ))
    elif args.cmd == "run":
        ids = [int(x) for x in args.ids.split(",")] if args.ids else None
        run_mod.run(
            args.records,
            config=_config(args),
            ids=ids,
            limit=args.limit,
            n_workers=args.workers,
            resume=args.resume,
            skip_smoke=args.skip_smoke,
            note=args.note,
        )
    elif args.cmd == "retry":
        run_mod.retry(
            args.run,
            args.records,
            config=_config(args),
            n_workers=args.workers,
            skip_smoke=args.skip_smoke,
            note=args.note,
        )
    elif args.cmd == "score":
        print(json.dumps(score(store.read_rows(args.run), _gold(args.gold)), indent=2))
    elif args.cmd == "stats":
        print(json.dumps(run_mod.stats(args.run), indent=2))
    elif args.cmd == "compare":
        gold = _gold(args.gold)
        print(compare(
            args.run_a,
            args.run_b,
            score(store.read_rows(args.run_a), gold),
            score(store.read_rows(args.run_b), gold),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
