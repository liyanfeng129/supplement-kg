# 0011.04 Human sheet: a local web page for the human's blind verdicts on the pilot

- **Status:** agreed and built (2026-10-27); the page is opened from notebook 16
- **Date:** 2026-10-27
- **Part of:** [0011 overview](00-overview.md); answers [03](03-rules.md#the-human-sheet-proposed-q20)'s
  Q20 with the web page. Built before the judges: it needs none of them.
- **Scope:** new `engine/repositories/prompts/extraction_judge/rulebook.md` and
  `load_rulebook` in `engine/repositories/prompts/loader.py`; additions to
  `engine/repositories/dsld/runs/` (`schema.py`, `paths.py`, `store.py`, `loader.py`); new
  `engine/services/extraction_eval/human_sheet.py` with additions to its `schema.py`; new
  frontend `web/human_sheet/` (`server.py`, `index.html`); notebook 16 redone; new
  `tests/test_extraction_eval_human_sheet.py`.

Decisions are tagged as in the overview: **[agreed]**, **[agreed 2026-10-27]**, **[open]**. Everything
below was agreed on 2026-10-27, when the user asked for it to be built and opened from the
notebook.

## Goal

The human gives the [03](03-rules.md) verdict on the 20 pilot labels flagged `in_human`
(2 per stratum), blind, in the same format the judges will use. The verdicts are stored under
the run's `eva/` folder, where the judges' verdicts and the metrics will later read them.
The 150-label human set uses the same sheet once the large set is drawn.

**Routed labels join the sheet [agreed 2026-10-27]:** the judges run first. Every other pilot
label on which they route anything to the human (03, "Routing to the human") is added to
the sheet, and the human gives it the same full verdict as the 20: every claim, and the
missed claims. Those labels are mixed into the one `sha256(case_id)` order, so the page does
not show which labels were routed. Their verdicts go to the same `human_pilot.jsonl`, and
the pilot's `in_human` flag tells the two apart afterwards.

## What the page shows **[agreed 2026-10-27]**

One label at a time, with the rulebook in a side panel:

- the **r4 passage**, as Qwen saw it, line by line;
- each of Qwen's claims as `P: <text>` or `R: <text>`, purposes first, in the row's order,
  not said to be Qwen's;
- per claim: `faithful` (`yes` · `invented` · `meaning_changed`), `actual_kind` (`purpose` ·
  `risk` · `neither` · `both`, disabled when `invented`), `unsure`, and an optional comment
  (`merged` dropped, `both` added, 2026-10-28: see 03);
- **missed claims**: select text in the passage, pick `purpose` or `risk`, add; each with
  `unsure` and an optional comment. The selection is the quote, so it cannot be mistyped;
- previous / next, a position (`3 of 20`) and which labels are saved;
- a timer of the time spent on the shown label, with a pause button **[agreed 2026-10-28]**.

**Blind:** no stratum, group, claim type, rank or model name is shown, and the labels come in
the order of `sha256(case_id)`, not the pilot's stratum order, which would give the strata
away.

**For the human, `rule` and `reason` are one optional comment [agreed 2026-10-27]:** 03 asks a judge for
a rule id, evidence and a reason with every verdict that is not correct. For the human, only
the closed fields are required, because only they are compared, and a non-expert reviewer's
free comment is more useful at adjudication than a forced rule id.

## Saving and checks **[agreed 2026-10-27]**

Save sends the whole label's verdict. The service checks it, and stores it only if it
passes; otherwise the page shows what is wrong:

- every claim of the label has a verdict, exactly once, and no other claim ids;
- every field is one of its values; `actual_kind` is empty exactly when `faithful` is
  `invented`; `active_seconds` is a whole number, not negative;
- every missed quote is a piece of the passage, and does not overlap one of the label's
  claims (it neither contains a claim's text nor lies inside it: a second claim inside a
  claim is part of that claim).

A label can be saved again: the latest save is the verdict.

**Timing [agreed 2026-10-28].** The pilot's time per label is measured (overview: the answer
key is timed). The timer runs only on the shown label, and stops on the pause button, while
the tab is hidden, and on a switch to another label, which starts its own. Each save stores
`active_seconds`, the label's active time since its previous save (since the page loaded, for
its first), and also when that stretch began and when it was saved (`opened_at`, `saved_at`,
for audit; their difference counts paused time and is not the measure). A label's time is the
sum of `active_seconds` over all its saves, so a correction counts too.

## Storage **[agreed 2026-10-27]**

```
data/dsld/runs/full214780-r4-2610131822-0/eva/
  human_pilot.jsonl     one HumanVerdict per line, appended on each save; the last line per case wins
```

Appended, not rewritten, so a crash mid-save loses at most that one save (a torn last line
is skipped on read), and every earlier save is kept as history. The human set's verdicts
later go to `human_set.jsonl` the same way. No manifest: what a verdict was given on is the
pilot, whose own manifest pins the rows and profile, and `read_pilot` raises if they changed.

## Interface **[agreed 2026-10-27]**

```python
# engine/repositories/dsld/runs/schema.py (additions)
Faithful = Literal["yes", "invented", "meaning_changed"]
ActualKind = Literal["purpose", "risk", "neither", "both"]

@dataclass(frozen=True)
class ClaimVerdict:
    claim_id: str                    # claim:<dsld_id>:<kind>:<i>
    faithful: Faithful
    actual_kind: ActualKind | None   # None exactly when invented
    unsure: bool
    comment: str                     # "" when none

@dataclass(frozen=True)
class MissedClaim:
    quote: str                       # a piece of the passage
    kind: Kind
    unsure: bool
    comment: str

@dataclass(frozen=True)
class HumanVerdict:
    case_id: str                     # label:<dsld_id>
    dsld_id: int
    claims: list[ClaimVerdict]       # purposes then risks, in the row's order
    missed: list[MissedClaim]
    opened_at: str                   # ISO 8601, UTC
    saved_at: str
    active_seconds: int              # active time since the label's previous save, pauses left out
```

```python
# engine/repositories/dsld/runs/paths.py (addition)
HUMAN_PILOT_NAME = "human_pilot.jsonl"

# engine/repositories/dsld/runs/store.py
def save_human_verdict(d: str | Path, verdict: HumanVerdict) -> None: ...   # appends one line

# engine/repositories/dsld/runs/loader.py
def read_human_verdicts(d: str | Path) -> dict[str, HumanVerdict]: ...     # last line per case_id
```

```python
# engine/repositories/prompts/loader.py (addition)
def load_rulebook(prompt_id: str) -> str: ...   # the text of <id>/rulebook.md
```

```python
# engine/services/extraction_eval/schema.py (additions)
@dataclass(frozen=True)
class SheetClaim:
    claim_id: str
    kind: Kind
    text: str

@dataclass(frozen=True)
class SheetCase:
    case_id: str
    dsld_id: int
    passage: str
    claims: list[SheetClaim]         # purposes then risks; no type, no stratum

# engine/services/extraction_eval/human_sheet.py
def sheet_cases(run_dir: str | Path, also: Iterable[str] = ()) -> list[SheetCase]: ...
def save(run_dir: str | Path, case: SheetCase, verdict: HumanVerdict) -> list[str]: ...
```

- `sheet_cases` reads the pilot (`read_pilot`), keeps the `in_human` cases and those whose
  case id is in `also` (raising on an id not in the pilot), joins each
  passage from the run's label set (refusing if its `passages_sha` is not the run
  manifest's), and orders them by `sha256(case_id)`.
- `save` runs the checks above against `case` and returns the problems; with none, it
  stores the verdict and returns `[]`. The case is passed back rather than read again,
  because a read joins the whole run and takes seconds.

```
# web/human_sheet/  (a frontend, beside engine/ as ADR 0002 §1 allows)
server.py     start(run_dir, port=8765, also=()) -> ThreadingHTTPServer, serving from a daemon thread
              (notebook 16 calls it); or python web/human_sheet/server.py <run_dir> [--port]
              stdlib http.server on 127.0.0.1: GET / the page, GET /api/cases the cases with
              their saved verdicts and the rulebook, POST /api/verdict -> save(), problems back
index.html    the page: plain HTML + JS, no build step, no CDN
```

Notebook 16 starts it and shows the page in an iframe, with a link as fallback; either goes
through VSCode's port forwarding of the SSH box. The server owns no workflow logic:
it turns HTTP into `sheet_cases` and `save` calls and back.

## The rulebook file **[agreed 2026-10-27; layout proposed, 03 Q21]**

`engine/repositories/prompts/extraction_judge/rulebook.md` is written in this step, since the
sheet's side panel shows it: Qwen's prompt in its own words with rule ids, without its output instructions and worked example, the decision rules, the
purpose line, and how to fill each field (03). The judge's system prompt is rendered from it
in the judge's spec. Until then it holds only what 03 has agreed.

## Notebook 16, redone **[agreed 2026-10-27]**

1. Setup, then the pilot draw (as now: read the stored pilot).
2. **Judges:** Claude and Codex judge all 50 pilot labels (inline in the notebook until the
   judge spec is agreed). Their agreement is counted, and `ROUTED` holds the labels outside
   the 20 that have anything routed.
3. **Human review:** a cell that starts the sheet with `also=ROUTED` and shows it inline, and
   a table of the sheet's labels with whether each is saved, its time (summed `active_seconds`), and counts of claims,
   flags and missed claims. No verdict content is shown while the review is in progress.

The Claude and Codex extraction cells are removed: under 03 they judge Qwen's extraction
rather than extracting their own. Their two runs under `data/dsld/runs/` stay on disk,
untouched.

## Acceptance

- `sheet_cases` on `full214780-r4-2610131822-0` returns the 20 `in_human` pilot labels, none
  of the other 30, in `sha256(case_id)` order; with `also`, those labels too, in the same
  order.
- `save` rejects each check's failure with a message naming it, and stores nothing.
- A saved label read back with `read_human_verdicts` equals what was saved; a second save of
  it wins; a torn last line is skipped.
- The page loads with nothing but the server running, saves, and shows saved labels after a
  reload.
- The timer counts on the shown label only, stops on pause, on a hidden tab and on a switch
  of label, and a save stores the time since the previous save.

## Open questions

03's Q21 (the rulebook's layout) is still proposed: `rulebook.md` follows it as written.

## The reveal: the judges beside the human's verdict **[built 2026-11-02]**

0011.03's blind adjudication (Q15), as its own page once the sheet is saved: new
`engine/services/extraction_eval/reveal.py`, `Adjudication` and `read_adjudications` /
`save_adjudication` in `engine/repositories/dsld/runs/`, new frontend `web/reveal/`, opened
from notebook 16 on port 8766.

An item is a place where the human and the judges do not all give the same answer, on a
saved label: every missed piece anyone listed (quotes where one holds the other share an item),
and every claim on which the three verdicts are not the same. The human decides each and may note
what the rulebook leaves open. A decision goes to `eva/adjudication_pilot.jsonl` with the blind
value beside it, and the last one on an item counts. `human_pilot.jsonl` is never changed:
agreement is measured on the blind verdicts.

The page can hide optional items on request: undecided missed pieces the human did not list and
only one judge did. The dots, the count and "next undecided" then skip them. They show by
default **[agreed 2026-11-03]**: on the pilot 14 of 30 such items became claims.

The final verdict is the decision where there is one, otherwise the blind verdict; agreement is
measured on the blind verdicts ([03](03-rules.md#who-judges-and-how-verdicts-become-final)).

## The pilot's verdicts, closed **[agreed 2026-11-03]**

The pilot is closed with 36 labels reviewed and all 72 reveal items decided. The verdicts stay
as saved, with no second review: it could not be blind now, and the pilot is not counted.

- **Rulebook:** a `HumanVerdict` holds no rulebook hash, so it is recorded here. 32 labels were
  saved up to 2026-11-02 10:30 UTC, before the last rulebook edit (16:44 UTC that day), on an earlier
  version; which one is not known, as the rulebook is not in git. The last 4 (`label:309467`,
  `label:286867`, `label:320121`, `label:62375`) were saved on 2026-11-03 under the frozen
  `aa9b5c22f611daa3`. The judges' verdicts in the reveal are all under it.
- **Known slips:** `claim:286867:risk:0` and `claim:286867:risk:1` are safety warnings saved as
  `yes/purpose`, by the human's later decision to `yes/risk` apparently mis-clicks. The blind
  record keeps them.
- **Missed claims:** the blind verdicts list 2 missed claims, both on one label, against 48
  from the judges. The human's blind review is not a list of everything missed.
