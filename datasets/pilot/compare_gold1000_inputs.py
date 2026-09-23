"""Measure field removal versus name removal with the local model tokenizer.

No inference or network calls. Frozen variants hold source text constant;
current_renderer additionally restores text omitted by the historical renderer.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.pipeline.config import Config
from engine.pipeline.extract import LocalExtractor
from engine.services.extraction.label_purpose_risk import label_text
from engine.services.sources.dsld import load_label


def variants(passage, current):
    header, marker, body = passage.partition("\nLABEL STATEMENTS:")
    if not marker:
        raise ValueError("Expected a frozen version-1 passage")
    body = body.lstrip("\n")
    statements = []
    for line in body.splitlines():
        match = re.fullmatch(r"\[([^\]]*)\] (.*)", line)
        if not match:
            raise ValueError(f"Unrecognised frozen statement: {line!r}")
        statements.append(match[2])
    text = "\n".join(statements)
    # PRODUCT/BRAND/etc. are field names; statement type strings are values.
    # Keep all values in this variant, including the type strings.
    header_values = []
    for line in header.splitlines():
        if line:
            key, separator, value = line.partition(": ")
            if not separator:
                raise ValueError(f"Unrecognised frozen header: {key!r}")
            header_values.append(value)
    return {
        "frozen": passage,
        "remove_header_names_only": "\n".join([*header_values, body]),
        "drop_metadata_fields": body,
        "drop_statement_type_fields": header + marker + ("\n" + text if text else ""),
        "drop_metadata_and_type_fields": text,
        "current_renderer": current,
    }


def compare(dump_root, model_path, fmt):
    from llama_cpp import Llama

    pilot = ROOT / "datasets/pilot"
    paths = [pilot / "gold-1000.jsonl", pilot / "gold-1000.sources.jsonl"]
    gold, sources = [[json.loads(line) for line in path.read_text().splitlines()
                      if line.strip()] for path in paths]
    ids = [row["dsld_id"] for row in gold]
    if len(ids) != 1000 or len(set(ids)) != 1000 or ids != [r["dsld_id"] for r in sources]:
        raise ValueError("Expected 1,000 matching, unique gold/source IDs in order")
    config = Config(model_id=str(model_path), fmt=fmt)
    renderer = LocalExtractor(config)
    model = Llama(model_path=str(model_path), vocab_only=True, n_gpu_layers=0, verbose=False)
    stats = {}
    try:
        for row, source in zip(gold, sources):
            current = label_text(load_label(row["dsld_id"], dump_root))
            for name, passage in variants(source["passage"], current).items():
                counts = stats.setdefault(name, {
                    "passage_tokens": 0, "full_prompt_tokens": 0,
                    "purpose_evidence_lost": 0, "risk_evidence_lost": 0,
                })
                counts["passage_tokens"] += len(model.tokenize(
                    passage.encode(), add_bos=False, special=False))
                counts["full_prompt_tokens"] += len(model.tokenize(
                    renderer.prompt_for(passage).encode(), add_bos=True, special=True))
                for kind in ("purpose", "risk"):
                    counts[kind + "_evidence_lost"] += sum(
                        not entry["evidence"] or entry["evidence"] not in passage
                        for entry in row[kind])
    finally:
        model.close()
    for counts in stats.values():
        for metric in ("passage_tokens", "full_prompt_tokens"):
            counts[metric + "_reduction_fraction"] = round(
                1 - counts[metric] / stats["frozen"][metric], 6)
    return {
        "labels": len(gold), "tokenizer_model": str(model_path),
        "prompt_format": fmt, "prompt_sha256": config.prompt_sha,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "variants": stats,
        "method": "Local GGUF tokenizer, vocabulary only; no inference. Passage counts exclude "
                  "BOS/chat/system tokens. Full prompt counts include the configured system prompt "
                  "and chat template. Frozen variants retain the same statement bodies; current_renderer "
                  "also restores excluded types and adds recursive ingredient free text.",
        "limits": "Evidence retention does not measure extraction accuracy. Header-name removal "
                  "retains metadata values; statement-type removal drops the type field's value too. "
                  "JSON keys such as notes were already absent from frozen passages.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, default=ROOT / "datasets/DSLD-full-database-JSON")
    parser.add_argument("--model", type=Path, default=Path(Config().model_id))
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    args = parser.parse_args()
    print(json.dumps(compare(args.dump_root, args.model, args.format), indent=2))
