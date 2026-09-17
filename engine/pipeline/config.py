"""Config identity: a hash over everything that changes the output.

    config_hash = sha256(canonical_json({
        prompt_id, prompt_sha, model_id, decode, schema_sha, renderer_version
    }))

Two runs pool only when their hash matches, so the hash must cover everything
that makes results incomparable and nothing that does not:

- `decode` records `enable_thinking: false` **explicitly**. This model ships
  with thinking on (`reasoning_effort: xhigh` by default in its chat template),
  and a silent flip would make every result incomparable with no other trace.
- `renderer_version` is hand-bumped beside the renderer, because editing what
  the model sees changes every output while prompt, model and params hold.
- The **git commit is not hashed** -- it goes in the manifest. Including it
  would give two genuinely identical runs different hashes a week apart and
  stop them pooling.

The resolved config is stored verbatim beside its hash, so a hash is always
invertible into the thing it names.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

from engine.pipeline.lines import LINES_VERSION
from engine.pipeline.render import RENDERER_VERSION
from engine.services.extraction.label_purpose_risk import SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[2]

# One editable file per prompt step, no version tree: most prompts are
# throwaway and iteration happens uncommitted, so numbered files accumulate
# dead copies and orphan every past run's prompt_sha once deleted. Each
# manifest embeds the prompt text verbatim instead.
DEFAULT_PROMPT = REPO_ROOT / "prompts" / "purpose_risk.md"

# The `lines` format's prompt. A separate file rather than a branch inside one:
# the two ask for different output shapes, and a prompt that described both
# would be worse at each.
LINES_PROMPT = REPO_ROOT / "prompts" / "purpose_risk_lines.md"

# How the model is made to produce structured output.
#
# - `json`: a GBNF grammar generated from SCHEMA constrains every token.
# - `lines`: `P: claim | 4` per claim, parsed by `engine.pipeline.lines`.
#
# `lines` exists because the grammar sampler costs 4.2x in decode speed and
# `evidence` roughly triples output tokens to restate text the passage already
# contains -- 233s vs 28s over ten labels. See `engine/pipeline/lines.py` for
# the measurement and for what replaces the grammar's guarantees.
FORMATS = ("json", "lines")

# The on-disk weights. An unextensioned blob in the HF cache, which is why
# earlier `*.gguf` searches found nothing.
#
# The GGUF header reports `general.file_type = 15`, which llama.cpp prints as
# "Q4_K - Medium", and the snapshot symlink names it UD-Q4_K_XL -- an Unsloth
# imatrix dynamic quant (1,251 chunks). ADR 0001 section 5 calls this Q6_K;
# that is a misreading of the same field and is corrected here. Nothing in the
# design depends on it: `smoke` measures verbatim grounding empirically rather
# than inferring copying fidelity from the quant level.
DEFAULT_MODEL = (
    "/var/tmp/ge74wan/hf_cache/hub/models--unsloth--Qwen3.8-27B-GGUF/blobs/"
    "3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj) -> str:
    """Stable JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def git_commit() -> str | None:
    """Current commit, for the manifest. Deliberately not part of the hash."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


@dataclass
class Decode:
    """Decode parameters. Every field here changes the output.

    `enable_thinking` is stored even though it is always False today: the
    point is that a future flip is visible in the hash rather than silent.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 2000
    seed: int = 0
    n_ctx: int = 8192
    enable_thinking: bool = False


@dataclass
class Config:
    """A resolved run configuration, hashable and invertible."""

    prompt_id: str = "purpose_risk"
    prompt_path: str | None = None
    model_id: str = DEFAULT_MODEL
    provenance: str = "local"
    decode: Decode = field(default_factory=Decode)
    note: str = ""
    fmt: str = "json"

    def __post_init__(self) -> None:
        if self.fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {self.fmt!r}")
        # The prompt follows the format unless the caller names a file. Pairing
        # them here means a `lines` run cannot silently send the JSON prompt,
        # which would produce zero parseable lines and look like a bad model.
        if self.prompt_path is None:
            self.prompt_path = str(LINES_PROMPT if self.fmt == "lines" else DEFAULT_PROMPT)

    @property
    def prompt(self) -> str:
        """The prompt text, read from its one editable file."""
        return Path(self.prompt_path).read_text()

    @property
    def prompt_sha(self) -> str:
        return _sha(self.prompt)

    @property
    def schema_sha(self) -> str:
        """The schema's sha under `json`; the parser's contract under `lines`.

        `lines` output is not schema-constrained, so hashing SCHEMA there would
        claim a guarantee that did not hold. Hashing the format name instead
        keeps the field meaningful: it still changes whenever the thing that
        shapes the output changes.
        """
        if self.fmt == "lines":
            return _sha(f"lines:{LINES_VERSION}")
        return _sha(canonical_json(SCHEMA))

    @property
    def identity(self) -> dict:
        """Exactly the fields the hash covers -- no more, no less.

        `fmt` is present only when it is not `json`. Adding a key
        unconditionally would have rehashed every past run: the hash is a
        promise that two runs pool, and changing what an unchanged
        configuration hashes to orphans the gold-100 baseline that new formats
        are supposed to be compared against. A `lines` run still gets its own
        hash -- via this key and via `schema_sha` -- which is the part that
        matters.
        """
        ident = {
            "prompt_id": self.prompt_id,
            "prompt_sha": self.prompt_sha,
            "model_id": self.model_id,
            "decode": asdict(self.decode),
            "schema_sha": self.schema_sha,
            "renderer_version": RENDERER_VERSION,
        }
        if self.fmt != "json":
            ident["fmt"] = self.fmt
        return ident

    @property
    def config_hash(self) -> str:
        return _sha(canonical_json(self.identity))

    def resolved(self) -> dict:
        """The blob stored verbatim beside the hash, so it stays invertible.

        Includes the prompt text in full: iteration happens uncommitted and
        dead prompt files get deleted, so a sha alone would not survive.
        """
        return {
            "config_hash": self.config_hash,
            **self.identity,
            "prompt": self.prompt,
            "prompt_path": self.prompt_path,
            "provenance": self.provenance,
            "note": self.note,
            # Stored verbatim so the hash stays invertible. Under `lines` there
            # is no schema; the manifest says so rather than embedding a schema
            # the run did not enforce.
            "schema": SCHEMA if self.fmt == "json" else None,
        }
