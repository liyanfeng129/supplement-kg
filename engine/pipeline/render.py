"""What the model sees: one label rendered to one passage.

The renderer is the single place deciding what reaches the model -- the
statement bodies and ingredient free text. That makes it
invisible to every other signal: editing it changes every output while prompt,
model and decode params are unchanged. So `RENDERER_VERSION` is hand-bumped
here and hashed into the config identity.

Bump it whenever `label_text()` changes what it emits. Hashing the function
source instead would churn on a comment edit; hashing its output would require
running it first.
"""

from __future__ import annotations

# Hand-bumped. 1 = the renderer as the pilot and gold set used it: three
# SKIP_TYPES, 25-ingredient cap, `[type] notes` lines, duplicate notes dropped.
# 2 = text-only statements across all types plus recursive ingredient free text.
RENDERER_VERSION = 2

# Re-export the shared renderer. Frozen pilot/gold passages remain version 1;
# regenerate separate inputs to evaluate version 2.
from engine.services.extraction.label_purpose_risk import (  # noqa: E402
    SKIP_TYPES,
    label_text,
)

__all__ = ["RENDERER_VERSION", "SKIP_TYPES", "label_text"]
