"""Compare two runs: the prompt diff above the metric delta.

A sha says *that* the prompt changed, the diff says *what* changed, and the
note says what you were trying to achieve. Printing the diff above the numbers
is the point: it makes you read what changed rather than only that something
changed.

Prompt text is read from each run's manifest, which embeds it verbatim --
iteration happens uncommitted and dead prompt files get deleted, so anything
relying on `git show` would have holes exactly during the phase that matters.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from engine.pipeline import store


def _flat(metrics: dict) -> dict:
    """Flatten the metrics worth showing a delta for."""
    p, e = metrics.get("presence", {}), metrics.get("evidence", {})
    return {
        "presence.accuracy": p.get("accuracy"),
        "hallucination_rate": metrics.get("hallucination_rate"),
        "evidence.precision": e.get("precision"),
        "evidence.recall": e.get("recall"),
        "n_labels": metrics.get("n_labels"),
    }


def compare(run_a: str | Path, run_b: str | Path, metrics_a: dict, metrics_b: dict) -> str:
    """Render the prompt diff, then the metric delta. Returns the text."""
    ma, mb = store.read_manifest(run_a), store.read_manifest(run_b)
    lines: list[str] = []

    lines.append(f"A: {Path(run_a).name}  note: {ma.get('note') or '(none)'}")
    lines.append(f"B: {Path(run_b).name}  note: {mb.get('note') or '(none)'}")
    lines.append("")

    pa, pb = ma.get("prompt", ""), mb.get("prompt", "")
    if pa == pb:
        lines.append(f"PROMPT: identical ({ma.get('prompt_sha', '')[:12]})")
    else:
        lines.append("PROMPT DIFF (A -> B):")
        lines.extend(
            difflib.unified_diff(
                pa.splitlines(), pb.splitlines(), fromfile="A", tofile="B", lineterm="", n=2
            )
        )
    if ma.get("config_hash") != mb.get("config_hash"):
        lines.append("")
        lines.append("NOTE: config hashes differ -- these runs do not pool.")
        for key in ("model_id", "renderer_version", "schema_sha"):
            if ma.get(key) != mb.get(key):
                lines.append(f"  {key}: {ma.get(key)} -> {mb.get(key)}")
        if ma.get("decode") != mb.get("decode"):
            lines.append(f"  decode: {ma.get('decode')} -> {mb.get('decode')}")

    lines.append("")
    lines.append("METRIC DELTA (B - A):")
    fa, fb = _flat(metrics_a), _flat(metrics_b)
    for key, va in fa.items():
        vb = fb.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            lines.append(f"  {key:24s} {va:>8.3f} -> {vb:>8.3f}  ({vb - va:+.3f})")
        else:
            lines.append(f"  {key:24s} {va} -> {vb}")

    lines.append("")
    lines.append(metrics_b.get("caveat", ""))
    return "\n".join(lines)
