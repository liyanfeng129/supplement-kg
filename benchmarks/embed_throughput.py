"""Which `n_servers`, `batch` and `n_ctx` embed DSLD passages fastest?

Spec 0007, acceptance 13; spec 0008, acceptance 9. The same seeded sample of real v4 passages, with the
`label_clustering` instruction, is embedded once per config through
`EmbedController` over `EmbedModel` -- the path a real run takes.

**Only the stream is timed.** `ready()` (load, end token, the three vector
checks) runs before the clock starts and is reported as `load_s`. `wall_s` is
first text pulled to last vector back; `tok_s` is the model's own prompt-token
count over it.

**`in_flight` follows the rule in `configs/embed.toml`:** the smallest count
with `in_flight x batch > n_servers`, plus one, so a freed server always finds a
request waiting. It is set per config, not swept.

**Peak VRAM** is nvidia-smi's `memory.used`, sampled every 0.2 s from just
before `ready()` to the end of the stream, so it includes the weights and the
compute buffer, all servers together. **Server errors** are the lines in the
servers' logs that mention an error, a failure or the KV cache being full; they
are counted per config and the first few kept, because a too-small `n_ctx`
shows up there before it shows up as a failed text.

**Isolation.** Refuses to start while any process holds the GPU, and waits for
the card to empty between configs. The model runs its servers itself, so one
copy of this script is the whole measurement.

    .venv/bin/python benchmarks/embed_throughput.py                  # 2000 texts, 1 2 3 servers x batch 16 32 64
    .venv/bin/python benchmarks/embed_throughput.py --servers 3 --batch 16 32 --ctx 4096 8192
    .venv/bin/python benchmarks/embed_throughput.py --n 200 --servers 3   # smoke

Results land in `data/benchmarks/embed-<run id>/`: `results.jsonl`, one row
per config as it finishes, and `manifest.json`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine import config  # noqa: E402
from engine.controllers.embed_controller import EmbedController  # noqa: E402
from engine.models.embed_model import EmbedModel  # noqa: E402
from engine.repositories.dsld.labels import loader as label_sets  # noqa: E402
from engine.repositories.prompts import loader as prompts  # noqa: E402

TEXT_SET = "full214780-r4"
INSTRUCTION_ID = "label_clustering"
RESULTS_ROOT = REPO_ROOT / "data/benchmarks"

# A server log line worth counting: an error, a failure, or a full KV cache.
SERVER_TROUBLE = re.compile(r"error|fail|no free|kv cache full|failed to find", re.IGNORECASE)


def gpu_holders() -> list[dict]:
    """Processes currently holding GPU memory, via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    holders = []
    for line in out.stdout.strip().splitlines() if out.returncode == 0 else []:
        parts = [f.strip() for f in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            holders.append({"pid": parts[0], "name": parts[1], "used_mib": parts[2]})
    return holders


def gpu_used_mib() -> int | None:
    """The card's `memory.used` in MiB, or None if nvidia-smi cannot say."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def run_config(texts: list[str], instruction: str, n_servers: int, batch: int, n_ctx: int,
               ubatch: int | None = None) -> dict:
    """Embed `texts` once under one config; the row for `results.jsonl`."""
    in_flight = n_servers // batch + 2
    row = {"n_servers": n_servers, "batch": batch, "in_flight": in_flight, "n_ctx": n_ctx, "ubatch": ubatch}

    peak = [gpu_used_mib() or 0]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.2):
            used = gpu_used_mib()
            if used is not None:
                peak[0] = max(peak[0], used)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    model = EmbedModel(n_servers=n_servers, n_ctx=n_ctx, ubatch=ubatch)
    ctl = EmbedController(model=model, instruction=instruction, batch=batch, in_flight=in_flight)
    try:
        started = time.monotonic()
        model.ready()
        row["load_s"] = round(time.monotonic() - started, 1)

        ok, failed = 0, {}
        meter = {}

        def on_progress(progress) -> None:
            meter["detail"] = progress.detail

        started = time.monotonic()
        for i, outcome in ctl.stream(enumerate(texts), total=len(texts), on_progress=on_progress):
            if outcome.ok:
                ok += 1
            else:
                failed[outcome.failed] = failed.get(outcome.failed, 0) + 1
        wall = time.monotonic() - started
        # The controller counts the model's prompt tokens; read them off its last report.
        match = re.search(r"(\d+) tok\b", meter.get("detail", ""))
        tokens = int(match.group(1)) if match else 0
        row.update({
            "wall_s": round(wall, 1),
            "tokens": tokens,
            "tok_s": round(tokens / wall) if wall else None,
            "texts_s": round(len(texts) / wall, 1) if wall else None,
            "ok": ok,
            "failed": failed,
        })
    except Exception as exc:  # a config that cannot run is a result, not the end of the sweep
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        log_paths = list(model.log_paths)
        ctl.close()
        stop.set()
        sampler.join()
        row["peak_vram_mib"] = peak[0]

    trouble = []
    for log_path in log_paths:
        if Path(log_path).exists():
            trouble += [line for line in Path(log_path).read_text(errors="replace").splitlines()
                        if SERVER_TROUBLE.search(line)]
    row["server_trouble_lines"] = len(trouble)
    row["server_trouble_sample"] = trouble[:5]
    row["server_logs"] = [str(p) for p in log_paths]
    return row


def wait_for_empty_gpu(timeout_s: float = 120) -> None:
    """Block until nothing holds the GPU; raise if it never empties."""
    deadline = time.monotonic() + timeout_s
    while gpu_holders():
        if time.monotonic() > deadline:
            raise SystemExit(f"GPU still held after {timeout_s:.0f}s: {gpu_holders()}")
        time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2000, help="passages to embed per config")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--servers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--batch", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--ctx", type=int, nargs="+", default=None,
                    help="n_ctx values; default the one in configs/embed.toml")
    ap.add_argument("--ubatch", type=int, default=None,
                    help="one forward pass, and the longest text; default the one in configs/embed.toml")
    args = ap.parse_args()

    holders = gpu_holders()
    if holders:
        raise SystemExit(f"Refusing to start: the GPU is held by {holders}")

    model_cfg = config.read_toml(config.EMBED_CONFIG_PATH).get("model", {})
    ctx_values = args.ctx or [model_cfg.get("n_ctx", 32768)]

    passages_sha, labels = label_sets.load_flattened(TEXT_SET)
    pool = sorted(labels)
    sample_ids = random.Random(args.seed).sample(pool, args.n)
    texts = [labels[i].passage for i in sample_ids]
    del labels, pool
    instruction = prompts.load_instruction(INSTRUCTION_ID)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / f"embed-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "text_set": TEXT_SET,
        "passages_sha": passages_sha,
        "n": args.n,
        "seed": args.seed,
        "instruction": instruction,
        "model": model_cfg,
        "grid": {"servers": args.servers, "batch": args.batch, "ctx": ctx_values, "ubatch": args.ubatch},
    }, indent=2))
    print(f"{args.n} passages, {sum(map(len, texts)):,} chars -> {out_dir}", flush=True)

    rows = []
    for n_ctx, n_servers, batch in itertools.product(ctx_values, args.servers, args.batch):
        wait_for_empty_gpu()
        row = run_config(texts, instruction, n_servers, batch, n_ctx, args.ubatch)
        rows.append(row)
        with open(out_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"servers={n_servers:<2} batch={batch:<3} in_flight={row['in_flight']:<2} ctx={n_ctx:<6} ub={args.ubatch or '-'} "
            + (f"ERROR {row['error']}" if "error" in row else
               f"load {row['load_s']:>5}s  wall {row['wall_s']:>6}s  {row['tok_s']:>6} tok/s  "
               f"failed {row['failed'] or 0}")
            + f"  peak {row['peak_vram_mib']} MiB  server trouble {row['server_trouble_lines']}",
            flush=True,
        )

    done = [r for r in rows if "error" not in r and not r["failed"]]
    if done:
        best = max(done, key=lambda r: r["tok_s"])
        print(f"\nfastest clean: servers={best['n_servers']} batch={best['batch']} "
              f"ctx={best['n_ctx']}  {best['tok_s']} tok/s  peak {best['peak_vram_mib']} MiB")


if __name__ == "__main__":
    main()
