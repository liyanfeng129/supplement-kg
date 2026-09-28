"""Does running Qwen with parallel slots finish a batch sooner?

The same labels, the same rendered prompts, greedy decoding, run through:

- `inprocess`  -- `QwenModel`, one generation at a time. Today's path.
- `server-N`   -- `llama-server -np N`, with N requests in flight at once, one
                  config per N in `--slots`. A slot is refilled the moment it
                  frees: a continuous pool.
- `controller` -- (`--controller`) `QwenController` over `QwenModel` as
                  configured: the model plans batches of its `n_parallel`
                  slots, and each batch finishes before the next starts. Set
                  beside `server-N` for the same N, this prices the batch
                  barrier.

`inprocess` runs `QwenModel.generate` one prompt at a time. Since `QwenModel`
moved onto `llama-server` it is no longer llama_cpp in process; the in-process
numbers are those of runs up to parallel-20260923T151057Z.

**Only the batch is timed.** Loading the weights, the hello check and one
warm-up request happen before the clock starts and are reported separately
as `load_s`. The number that answers the question is `wall_s`: first request
sent to last answer back.

**Speed is worthless if the output rots.** This architecture has a known CUDA
failure that loads cleanly and emits fluent garbage, and `llama-server` links
a different ggml build from the one in the venv. So every config is also
scored for grounding (a fluent fabrication cannot reproduce exact substrings of
the label), for parse status, and for exact-match against the in-process text.
Batching can legitimately change greedy output a little, since floating-point
sums run in a different order; a fall in grounding cannot legitimately happen.

**Slot size is measured, not guessed.** Every rendered prompt is tokenised
with the model's own vocabulary before anything loads; one slot holds the
longest prompt plus the output budget, and `-c` is that times N with the KV
cache split per slot (`--no-kv-unified`), so every slot can hold any label.

**Isolation.** Refuses to start while any process holds the GPU -- numbers
taken beside another model are worse than none -- and waits for the card to
empty between configs.

    .venv/bin/python benchmarks/parallel_slots.py                   # 32 labels, 1 2 4 8
    .venv/bin/python benchmarks/parallel_slots.py --n 64 --slots 1 4 8 --format json
    .venv/bin/python benchmarks/parallel_slots.py --slots --skip-inprocess \
        --variant "base=" --variant "ub1024=-ub 1024" --variant "primed=@prime"

A `--variant NAME=ARGS` runs one more server config at `--variant-slots`
slots with ARGS added to its command line. The token `@prime` is not a flag:
it sends every slot the prompts' shared prefix (the system turn) before the
clock starts, so each slot holds it in cache from its first real request.

Results land in `data/benchmarks/parallel-<run id>/`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GGML_CUDA_NO_PINNED", "1")

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.models.base_model import Prompt  # noqa: E402
from engine.models.qwen_model import (  # noqa: E402
    HELLO_PROMPT,
    QwenModel,
    gpu_holders,
    incoherence,
)
from engine.pipeline import lines  # noqa: E402
from engine.pipeline.validate import grounding_rate, validate  # noqa: E402
from engine.services.extraction.label_purpose_risk import SCHEMA  # noqa: E402

LLAMA_SERVER = Path("/var/tmp/ge74wan/llama.cpp/build/bin/llama-server")
DEFAULT_INPUTS = REPO_ROOT / "data/dsld/runs/silver1000-text-v2/raw/inputs.jsonl"
RESULTS_ROOT = REPO_ROOT / "data/benchmarks"

# Per format: the system prompt the pipeline uses, how generation stops, the
# output budget, and whether a schema constrains it.
FORMATS = {
    "lines": {
        "system": REPO_ROOT / "prompts/purpose_risk_lines.md",
        "stop": tuple(lines.STOP),
        "max_tokens": lines.MAX_TOKENS,
        "schema": None,
    },
    "json": {
        "system": REPO_ROOT / "prompts/purpose_risk.md",
        "stop": ("<|im_end|>",),
        "max_tokens": 2000,
        "schema": SCHEMA,
    },
}


# --- inputs -----------------------------------------------------------------


def load_labels(path: Path, n: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    return sorted(random.Random(seed).sample(rows, n), key=lambda r: r["dsld_id"])


def build_prompts(labels: list[dict], fmt: str) -> list[Prompt]:
    spec = FORMATS[fmt]
    system = spec["system"].read_text()
    return [
        Prompt(
            system=system,
            user=label["passage"],
            schema=spec["schema"],
            stop=spec["stop"],
            max_tokens=spec["max_tokens"],
        )
        for label in labels
    ]


def slot_ctx_for(rendered: list[str], max_tokens: int, model_path: str) -> dict:
    """Tokens one slot must hold: the longest prompt plus the output budget."""
    from llama_cpp import Llama

    vocab = Llama(model_path=model_path, vocab_only=True, verbose=False)
    counts = [len(vocab.tokenize(r.encode(), add_bos=False, special=True)) for r in rendered]
    need = max(counts) + max_tokens
    return {
        "prompt_tokens_max": max(counts),
        "prompt_tokens_median": int(statistics.median(counts)),
        "max_tokens": max_tokens,
        "slot_ctx": -(-need // 256) * 256,
    }


# --- the GPU ----------------------------------------------------------------


def gpu_sample() -> tuple[int, int] | None:
    """(memory used MiB, utilisation %) on device 0, or None when unreadable."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    first = out.stdout.strip().splitlines()[:1]
    try:
        used, util = (int(f) for f in first[0].split(","))
    except (IndexError, ValueError):
        return None
    return used, util


# MiB another process may hold and still be tolerated: an idle CUDA context
# (a notebook kernel that closed its model keeps ~300 MiB) competes for no
# compute. Anything model-sized is refused. Set from `--tolerate-mib`.
TOLERATE_MIB = 0


def require_free_gpu(wait_s: float = 0) -> list[dict]:
    """Refuse to measure beside anything else on the card; return what was tolerated."""
    deadline = time.monotonic() + wait_s
    while True:
        holders = gpu_holders()
        held = [h for h in holders if not (h["used_mib"].isdigit() and int(h["used_mib"]) <= TOLERATE_MIB)]
        if not held:
            return holders
        if time.monotonic() >= deadline:
            names = ", ".join(f"PID {h['pid']} {h['name']} ({h['used_mib']} MiB)" for h in held)
            sys.exit(
                f"The GPU is held by: {names}.\n"
                "Stop it (a notebook kernel with the model loaded counts) and rerun: "
                "a benchmark beside another process measures the contention, not the model."
            )
        time.sleep(2)


class PeakVram:
    """Samples the GPU in the background: `.peak` VRAM and `.util` percentages.

    Utilisation is nvidia-smi's "a kernel was running" share of each sample
    window -- coarse, it cannot see how full each kernel kept the card, but a
    value well under 100 does mean the GPU sat idle between steps.
    """

    def __init__(self, every_s: float = 0.5):
        self.peak = 0
        self.util: list[int] = []
        self._every = every_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            sample = gpu_sample()
            if sample:
                self.peak = max(self.peak, sample[0])
                self.util.append(sample[1])
            self._stop.wait(self._every)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# --- in-process -------------------------------------------------------------


def run_inprocess(prompts: list[Prompt]) -> dict:
    """Today's path. Runs in a child process so the weights leave with it."""
    model = QwenModel()
    started = time.monotonic()
    model.ready()  # load + hello check: excluded from the timed batch
    load_s = time.monotonic() - started

    outputs = []
    with PeakVram() as vram:
        started = time.monotonic()
        for prompt in prompts:
            g = model.generate(prompt)
            outputs.append(
                {
                    "text": g.text,
                    "latency_ms": g.latency_ms,
                    "prompt_tokens": g.usage.prompt_tokens,
                    "completion_tokens": g.usage.completion_tokens,
                    "stop_reason": g.stop_reason,
                }
            )
        wall_s = time.monotonic() - started
    model.close()
    return {
        "load_s": load_s,
        "wall_s": wall_s,
        "peak_vram_mib": vram.peak,
        "gpu_util_mean": _mean(vram.util),
        "outputs": outputs,
    }


def _mean(values: list[int]) -> float | None:
    return round(statistics.mean(values), 1) if values else None


def run_controller(prompts: list[Prompt]) -> dict:
    """The engine's own path: the model plans batches, the controller runs them."""
    from engine.controllers.qwen_controller import QwenController

    model = QwenModel()
    require_free_gpu(wait_s=120)
    started = time.monotonic()
    model.ready()
    load_s = time.monotonic() - started
    try:
        with PeakVram() as vram:
            started = time.monotonic()
            outcomes = QwenController(model).handle(prompts)
            wall_s = time.monotonic() - started
    finally:
        model.close()
    return {
        "load_s": load_s,
        "wall_s": wall_s,
        "peak_vram_mib": vram.peak,
        "gpu_util_mean": _mean(vram.util),
        "outputs": [
            {
                "text": o.generation.text,
                "latency_ms": o.generation.latency_ms,
                "prompt_tokens": o.generation.usage.prompt_tokens,
                "completion_tokens": o.generation.usage.completion_tokens,
                "stop_reason": o.generation.stop_reason,
            }
            for o in outcomes
        ],
    }


def run_inprocess_in_child(prompts_path: Path, out_path: Path) -> dict:
    subprocess.run(
        [sys.executable, __file__, "--_inprocess", str(prompts_path), str(out_path)],
        check=True,
    )
    return json.loads(out_path.read_text())


# --- llama-server -----------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(n_slots: int, slot_ctx: int, model: QwenModel, log_path: Path, extra=()):
    port = free_port()
    args = [
        str(LLAMA_SERVER),
        "-m", model.model_path,
        "-ngl", "999",
        "-np", str(n_slots),
        "-c", str(n_slots * slot_ctx),
        "--no-kv-unified",
        "-s", str(model.seed),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-webui",
        *extra,
    ]
    log = log_path.open("w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    return proc, f"http://127.0.0.1:{port}", args


def wait_healthy(proc, base: str, timeout_s: float = 900) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode} while loading")
        try:
            if requests.get(f"{base}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become healthy in time")


def complete(base: str, model: QwenModel, prompt: Prompt) -> dict:
    body = {
        "prompt": model._render(prompt),  # byte-identical to what QwenModel sends
        "n_predict": prompt.max_tokens or model.max_tokens,
        "temperature": model.temperature,
        "top_p": model.top_p,
        "top_k": model.top_k,
        "seed": model.seed,
        "stop": list(prompt.stop) or ["<|im_end|>"],
        "cache_prompt": True,
    }
    if prompt.schema is not None:
        body["json_schema"] = prompt.schema
    started = time.monotonic()
    r = requests.post(f"{base}/completion", json=body, timeout=1800)
    r.raise_for_status()
    out = r.json()
    t = out.get("timings") or {}
    return {
        "text": out.get("content") or "",
        "latency_ms": int((time.monotonic() - started) * 1000),
        "prompt_tokens": out.get("tokens_evaluated") or 0,
        "completion_tokens": out.get("tokens_predicted") or 0,
        "cached_tokens": t.get("cache_n", out.get("tokens_cached") or 0),
        "stop_reason": out.get("stop_type") or "",
    }


def shared_prefix(model: QwenModel, prompts: list[Prompt]) -> str:
    """The rendered text every prompt starts with, cut after the last full turn.

    Cut at a turn boundary (`<|im_end|>\n`) so the prefix tokenises exactly
    as it does inside each full prompt; a cut mid-text could merge
    differently with what follows.
    """
    common = os.path.commonprefix([model._render(p) for p in prompts])
    end = common.rfind("<|im_end|>\n")
    return common[: end + len("<|im_end|>\n")] if end >= 0 else ""


def prime_slots(base: str, n_slots: int, prefix: str) -> None:
    """Have every slot read the shared prefix once, untimed."""
    body = {"prompt": prefix, "n_predict": 1, "temperature": 0, "cache_prompt": True}
    with ThreadPoolExecutor(max_workers=n_slots) as pool:
        list(pool.map(lambda _: requests.post(f"{base}/completion", json=body, timeout=600), range(n_slots)))


def run_server(
    n_slots: int, slot_ctx: int, prompts: list[Prompt], out_dir: Path, name: str | None = None, extra=(), prime=False
) -> dict:
    model = QwenModel()  # construction only: settings and the renderer
    require_free_gpu(wait_s=120)
    name = name or f"server-{n_slots}"
    proc, base, args = start_server(n_slots, slot_ctx, model, out_dir / f"{name}.log", extra)
    try:
        started = time.monotonic()
        wait_healthy(proc, base)
        hello = complete(base, model, HELLO_PROMPT)["text"]  # warm-up, untimed
        why = incoherence(hello)
        if why:
            raise RuntimeError(f"{name} failed the hello check ({why}): {hello[:200]!r}")
        if prime:
            prime_slots(base, n_slots, shared_prefix(model, prompts))
        load_s = time.monotonic() - started

        with PeakVram() as vram:
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=n_slots) as pool:
                outputs = list(pool.map(lambda p: complete(base, model, p), prompts))
            wall_s = time.monotonic() - started
        return {
            "load_s": load_s,
            "wall_s": wall_s,
            "peak_vram_mib": vram.peak,
            "gpu_util_mean": _mean(vram.util),
            "command": args,
            "outputs": outputs,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# --- scoring ----------------------------------------------------------------


def score(result: dict, labels: list[dict], fmt: str, reference: list[str] | None) -> dict:
    outs = result["outputs"]
    rows = [
        validate(o["text"], label["passage"], label["dsld_id"], fmt=fmt)
        for o, label in zip(outs, labels)
    ]
    completion = sum(o["completion_tokens"] for o in outs)
    read = [o["prompt_tokens"] - o.get("cached_tokens", 0) for o in outs if "cached_tokens" in o]
    s = {
        "wall_s": round(result["wall_s"], 1),
        "labels_per_min": round(len(outs) / result["wall_s"] * 60, 2),
        "completion_tok_per_s": round(completion / result["wall_s"], 1),
        "median_latency_s": round(statistics.median(o["latency_ms"] for o in outs) / 1000, 2),
        "load_s_excluded": round(result["load_s"], 1),
        "peak_vram_mib": result["peak_vram_mib"],
        "gpu_util_mean": result.get("gpu_util_mean"),
        "grounding": grounding_rate(rows),
        "statuses": {st: sum(r["status"] == st for r in rows) for st in {r["status"] for r in rows}},
        "hit_max_tokens": sum(o["stop_reason"] in ("length", "limit") for o in outs),
        "prompt_read_per_label": round(statistics.mean(read)) if read else None,
    }
    if reference is not None:
        texts = [o["text"] for o in outs]
        s["exact_match_vs_inprocess"] = round(
            sum(a == b for a, b in zip(texts, reference)) / len(texts), 3
        )
        s["mean_similarity_vs_inprocess"] = round(
            statistics.mean(difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(texts, reference)), 3
        )
    return s


def print_table(summary: dict) -> None:
    base_name = next((n for n in ("server-1", "inprocess") if n in summary), next(iter(summary)))
    base = summary[base_name]
    cols = {
        "wall_s": "wall s",
        "labels_per_min": "labels/min",
        "completion_tok_per_s": "out tok/s",
        "peak_vram_mib": "peak MiB",
        "gpu_util_mean": "gpu util %",
        "prompt_read_per_label": "read/label",
        "grounding": "grounding",
        "exact_match_vs_inprocess": "same text",
    }
    print("\n" + "config".ljust(12) + "".join(h.rjust(12) for h in cols.values()) + "speedup".rjust(10))
    for name, s in summary.items():
        speedup = base["wall_s"] / s["wall_s"]
        cells = "".join(
            (f"{s[c]:.3g}" if isinstance(s.get(c), float) else str(s.get(c, "-"))).rjust(12) for c in cols
        )
        print(name.ljust(12) + cells + f"{speedup:9.2f}x")
    print(f"\nspeedup is against {base_name}; load time is excluded throughout.")


# --- main -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=32, help="labels in the batch")
    ap.add_argument("--slots", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--format", choices=sorted(FORMATS), default="lines")
    ap.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--seed", type=int, default=20260923, help="which labels are drawn")
    ap.add_argument("--skip-inprocess", action="store_true")
    ap.add_argument("--controller", action="store_true", help="also run QwenController over QwenModel")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=ARGS",
                    help="one more server config: extra llama-server args, @prime to prime slots")
    ap.add_argument("--variant-slots", type=int, default=4)
    ap.add_argument(
        "--tolerate-mib", type=int, default=0,
        help="allow other GPU processes holding at most this many MiB (idle CUDA contexts)",
    )
    ap.add_argument("--_inprocess", nargs=2, type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._inprocess:  # child mode: run the in-process baseline and exit
        prompts = [Prompt(**{**p, "stop": tuple(p["stop"])}) for p in json.loads(a._inprocess[0].read_text())]
        a._inprocess[1].write_text(json.dumps(run_inprocess(prompts)))
        return

    if not LLAMA_SERVER.exists():
        sys.exit(f"llama-server not found at {LLAMA_SERVER}")
    global TOLERATE_MIB
    TOLERATE_MIB = a.tolerate_mib
    tolerated = require_free_gpu()
    if tolerated:
        print(f"tolerating idle GPU holders: {tolerated}", flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / f"parallel-{run_id}"
    out_dir.mkdir(parents=True)

    labels = load_labels(a.inputs, a.n, a.seed)
    prompts = build_prompts(labels, a.format)
    model = QwenModel()
    sizing = slot_ctx_for([model._render(p) for p in prompts], FORMATS[a.format]["max_tokens"], model.model_path)
    print(f"{len(labels)} labels, format={a.format}, sizing={sizing}", flush=True)

    results: dict[str, dict] = {}
    if not a.skip_inprocess:
        print("inprocess ...", flush=True)
        prompts_path = out_dir / "prompts.json"
        prompts_path.write_text(
            json.dumps([{"system": p.system, "user": p.user, "schema": p.schema, "stop": list(p.stop), "max_tokens": p.max_tokens} for p in prompts])
        )
        results["inprocess"] = run_inprocess_in_child(prompts_path, out_dir / "inprocess.json")

    for n_slots in a.slots:
        print(f"server-{n_slots} ...", flush=True)
        results[f"server-{n_slots}"] = run_server(n_slots, sizing["slot_ctx"], prompts, out_dir)

    for spec in a.variant:
        name, _, extra = spec.partition("=")
        tokens = shlex.split(extra)
        prime = "@prime" in tokens
        tokens = [t for t in tokens if t != "@prime"]
        print(f"{name} ({a.variant_slots} slots, {' '.join(tokens) or 'defaults'}{', primed' if prime else ''}) ...", flush=True)
        try:
            results[name] = run_server(a.variant_slots, sizing["slot_ctx"], prompts, out_dir, name, tokens, prime)
        except Exception as exc:  # one broken flag must not cost the rest of a sweep
            print(f"  {name} failed: {exc} (see {out_dir / (name + '.log')})", flush=True)

    if a.controller:
        print("controller ...", flush=True)
        results["controller"] = run_controller(prompts)

    first = next(iter(results.values()), None)
    reference = [o["text"] for o in first["outputs"]] if first else None
    summary = {name: score(r, labels, a.format, reference) for name, r in results.items()}

    manifest = {
        "run_id": run_id,
        "format": a.format,
        "n": len(labels),
        "seed": a.seed,
        "inputs": str(a.inputs),
        "dsld_ids": [label["dsld_id"] for label in labels],
        "sizing": sizing,
        "model_path": model.model_path,
        "llama_server": str(LLAMA_SERVER),
        "server_commands": {k: r["command"] for k, r in results.items() if "command" in r},
        "tolerated_gpu_holders": tolerated,
        "summary": summary,
    }
    (out_dir / "results.json").write_text(json.dumps(manifest, indent=2))
    with (out_dir / "outputs.jsonl").open("w") as f:
        for name, r in results.items():
            for label, o in zip(labels, r["outputs"]):
                f.write(json.dumps({"config": name, "dsld_id": label["dsld_id"], **o}) + "\n")

    print_table(summary)
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
