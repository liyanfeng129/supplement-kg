"""The `lines` output format: `P: claim | 4`, parsed back to the row shape.

**Why a second format exists.** The GBNF grammar costs 4.2x in raw decode
speed -- 8.3 vs 34.6 tok/s, measured on the same ten labels with the same
prompt and the same output shape, the only difference being whether the
grammar sampler ran. llama.cpp evaluates the grammar against the whole vocab
at every token, and that is the bill. On top of that, `evidence` is the label
copied back verbatim, so it roughly triples output tokens to restate text the
passage already contains. Together: 233s -> 28s over ten labels.

**What replaces `evidence`.** Nothing, in the output -- the prompt instead
orders the model to copy the label's own wording, and `validate()` checks the
claim itself against the passage. That rule is load-bearing: without it the
same format scored 50/57 claims verbatim, with it 54/54. `recover_span()`
finds the sentence a claim came from when a reviewer needs the surrounding
context, and it does so by search, which is free.

**Why `END` is in the prompt.** A grammar also ends a generation: once the
JSON closes, no other token is legal. Free-form lines have no such stop, so a
model that does not emit `<|im_end|>` runs to `max_tokens` -- which is how a
benchmark that should take 75s ran past 560s. `END` is a stop sequence, not
decoration; removing it to save two tokens reintroduces the runaway.

Parsing is deliberately strict and counts what it rejects. A line the regex
does not match is a dropped claim, and a silent drop is indistinguishable from
a label that genuinely said nothing -- so `parse()` returns `stray`, `run`
aggregates it, and the manifest carries it.
"""

from __future__ import annotations

import re

# `P:`/`R:` over `PURPOSE:`/`RISK:` saves ~1 token per claim and measured
# identically (zero stray lines across 54 claims). The trade is legibility of
# the stored `raw` text months later, which `text` and `field` on the row make
# up for.
FIELDS = {"P": "purpose", "R": "risk"}

# Hand-bumped whenever the line grammar or `parse()` changes shape, exactly as
# `RENDERER_VERSION` is: `Config.schema_sha` hashes it, so a parser change that
# alters output stops old and new runs from pooling instead of silently mixing.
LINES_VERSION = 3

# One claim per line: field, text, confidence. The text is non-greedy so the
# final `| <digit>` binds to the last pipe, letting a claim contain a pipe.
LINE_RE = re.compile(r"^\s*(P|R)\s*:\s*(.+?)\s*\|\s*([0-5])\s*$", re.IGNORECASE)

# The model is told to end with END; the stop sequence usually eats it, but a
# trailing or bare END must never count as a stray line.
END_RE = re.compile(r"^\s*END\s*$", re.IGNORECASE)

STOP = ["<|im_end|>", "\nEND"]

# Enough for the longest healthy generation (186 tokens over the benchmark's
# ten labels) with room to spare, low enough that one runaway costs seconds
# rather than a minute. Overridden by `Decode.max_tokens` when that is lower.
MAX_TOKENS = 400


# Boilerplate risks, dropped after parsing rather than by prompt rule.
#
# Measured on gold-100: 33 of 38 false-positive claims on cells gold left empty
# were these two families -- child-safety (20) and package integrity (13). They
# are the difference between this format's 0.474 hallucination-given-empty and
# the grammar's 0.316.
#
# Why not a prompt rule. The child-safety rule is conditional on *all other*
# risks ("include it only if no other risk text exists"), and a format that
# emits one line at a time cannot evaluate that while generating -- the model
# would have to know what it is about to write. The parser sees every claim at
# once, so the condition is cheap here and unenforceable there. Package
# integrity is unconditional: a broken seal is not a health caution about the
# supplement, and gold never treats it as one.
#
# Matched on the normalised claim, so wording variants ("keep out of the reach
# of children") collapse together.
_CHILD_SAFETY = re.compile(r"\bkeep out of (the )?reach of children\b", re.IGNORECASE)
_PACKAGE_INTEGRITY = re.compile(
    r"\b(do not (use|eat)\b.*\b(seal|freshness packet)|tamper[- ]resistant|"
    r"(please )?discard the inedible freshness packet)",
    re.IGNORECASE,
)


def _is_boilerplate_risk(text: str) -> bool:
    return bool(_CHILD_SAFETY.search(text) or _PACKAGE_INTEGRITY.search(text))


def drop_boilerplate_risks(risks: list[dict]) -> list[dict]:
    """Drop child-safety and packaging boilerplate from a label's risks.

    The prompt keeps child-safety when no other risk text exists. Measured
    against gold-100, that carve-out is wrong far more often than it is right:
    gold treats "keep out of reach of children" as a risk in 6 of 100 labels,
    while an "only risk present" rule fires on 20. Keeping the carve-out left
    19 false positives standing -- the entire remaining gap to the grammar
    baseline on `hallucination_rate_given_empty`.

    So it is dropped unconditionally. The cost is those 6 labels, where the
    warning was the only caution and is now missing; the gain is 19 cells that
    stop asserting a risk the gold reading says is not one. That trade is
    deliberate and worth revisiting if the gold set is ever re-cut -- this
    boundary is a judgement about what counts as a risk, not a fact about the
    labels, and the grammar format gets it wrong in the same direction (it
    emitted 15).
    """
    return [r for r in risks if not _is_boilerplate_risk(r.get("text") or "")]


def parse(text: str, passage: str | None = None) -> dict:
    """Raw model text to `{purpose, risk, ..._confidence, stray}`.

    Confidence is per claim in the output but per field on the row, because
    that is the shape `validate()` and the gold set already use. The field
    score is the max over its claims: the row's confidence answers "how
    clearly does this label state a purpose at all", and one unambiguous claim
    settles that regardless of a second, vaguer one.

    Given `passage`, each claim's `evidence` is the sentence it was found in,
    recovered by search rather than generated -- the model never spends tokens
    restating the label. Without it, `evidence` falls back to the claim itself,
    which keeps the row shape valid for a caller that has no passage to hand.
    """
    out: dict = {
        "purpose": [],
        "risk": [],
        "purpose_confidence": 0,
        "risk_confidence": 0,
        "stray": 0,
        "stray_lines": [],
    }
    confs: dict[str, list[int]] = {"purpose": [], "risk": []}

    for line in (text or "").splitlines():
        if not line.strip() or END_RE.match(line):
            continue
        m = LINE_RE.match(line)
        if not m:
            out["stray"] += 1
            # Kept, bounded, for diagnosis: a format drift is only fixable if
            # the offending line survives to be read.
            if len(out["stray_lines"]) < 5:
                out["stray_lines"].append(line.strip()[:200])
            continue
        field = FIELDS[m.group(1).upper()]
        claim = m.group(2).strip().strip("*").strip()
        if not claim:
            out["stray"] += 1
            continue
        # `evidence` is the label sentence the claim came from, found by
        # search. The row shape stays identical to the grammar format's, so
        # `validate()` and every downstream reader keep working unchanged --
        # the difference is only that the span costs a regex here instead of
        # ~35 output tokens there.
        #
        # The claim itself is the fallback when recovery finds nothing. That
        # keeps `evidence` non-empty (an empty one fails the grounding check on
        # a technicality rather than on truthfulness) and leaves the check
        # meaningful: a fabricated claim is not in the passage, so it stays
        # ungrounded either way.
        evidence = claim
        if passage is not None:
            evidence = recover_span(claim, passage) or claim
        out[field].append({"text": claim, "evidence": evidence})
        confs[field].append(int(m.group(3)))

    # Confidence is computed before the filter and carried by the claims that
    # survive it, so dropping boilerplate cannot raise a field's score.
    kept_risk = drop_boilerplate_risks(out["risk"])
    if len(kept_risk) != len(out["risk"]):
        dropped = {id(r) for r in kept_risk}
        confs["risk"] = [
            c for r, c in zip(out["risk"], confs["risk"]) if id(r) in dropped
        ]
        out["risk"] = kept_risk

    for field in ("purpose", "risk"):
        if confs[field]:
            out[f"{field}_confidence"] = max(confs[field])
    return out


# A rendered label line starts a new statement when it opens with a
# "[Statement Type]" bracket or a "FIELD:" caps header; anything else is the
# previous statement continuing after a wrap.
_STATEMENT_START = re.compile(r"^\s*(\[[^\]]*\]|[A-Z][A-Z /]{2,}:)")


def _spans(passage: str) -> list[str]:
    """Candidate source spans: label statements, split into sentences.

    Statements are unwrapped before they are split, because the renderer wraps
    long ones across lines: splitting on newlines first would cut "Do not
    exceed recommended\\nintake." in half, and no sentence span would then
    contain the claim. The window fallback in `recover_span` would paper over
    that by joining unrelated lines into one blob -- which is worse than
    failing, since that blob becomes `evidence` and is not a clean substring of
    anything.
    """
    statements: list[str] = []
    for line in passage.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if statements and not _STATEMENT_START.match(line):
            statements[-1] += " " + stripped  # a wrapped continuation
        else:
            statements.append(stripped)

    spans = []
    for statement in statements:
        body = re.sub(r"^\[[^\]]*\]\s*", "", statement)  # drop the type prefix
        for part in re.split(r"(?<=[.!?])\s+", body):
            part = part.strip()
            if part:
                spans.append(part)
    return spans


def recover_span(text: str, passage: str) -> str | None:
    """The sentence a claim came from, found by search rather than generated.

    Replayed over the 515 entries of the last grammar run -- discarding each
    entry's model-written evidence and re-finding it from `text` alone -- this
    recovered a usable span for 508 (98.6%), exactly matching the model's own
    evidence for 329. That is what makes dropping `evidence` from the output
    safe: the span was never information only the model had.

    Returns None when the claim is not in the passage, which for a grounded
    row cannot happen and for an ungrounded one is the honest answer.
    """
    norm = " ".join((text or "").split()).lower()
    if not norm:
        return None
    spans = _spans(passage)

    hits = [s for s in spans if norm in " ".join(s.split()).lower()]
    if hits:
        return min(hits, key=len)

    # A claim can straddle a line break in the passage; widen a window until it
    # fits. Four spans covers every case seen in the 515-entry replay.
    for i in range(len(spans)):
        for j in range(i + 1, min(i + 4, len(spans)) + 1):
            window = " ".join(spans[i:j])
            if norm in " ".join(window.split()).lower():
                return window
    return None
