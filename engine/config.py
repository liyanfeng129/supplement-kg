"""Environment-backed settings.

Importing this module loads `.env` from the repo root, so `os.getenv` works
the same in a notebook, a script, and a test without each one wiring dotenv
itself. Real values live in `.env` (gitignored); `.env.example` documents the
keys.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

# override=False: an explicitly exported shell var beats the file, which is
# what you want on a cluster or in CI where secrets arrive as real env vars.
load_dotenv(ENV_PATH, override=False)


def get(name: str, default: str | None = None) -> str | None:
    """Read a setting, treating blank as unset.

    The `.env.example` template ships keys with empty values, so a copied-but-
    unfilled `.env` would otherwise hand back "" and look configured.
    """
    val = os.getenv(name, default)
    if val is not None:
        val = val.strip()
    return val or default


def require(name: str) -> str:
    """Read a setting that has no sensible default."""
    val = get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return val


NCBI_API_KEY = get("NCBI_API_KEY")
NCBI_EMAIL = get("NCBI_EMAIL")

# Anthropic API key, for LLM extraction of purpose/risk from DSLD label text.
ANTHROPIC_API_KEY = get("ANTHROPIC_API_KEY")

# DSLD (NIH ODS) API key, issued via api.data.gov. Optional: DSLD answers
# unauthenticated requests too, but at 1,000/hour/IP -- low enough that a
# per-label sweep is measured in days. See docs/dsld-purpose-risk-findings.md.
DSLD_API_KEY = get("DSLD_API_KEY")
