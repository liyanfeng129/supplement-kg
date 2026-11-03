"""The reveal: where the human and the judges differ on a reviewed label, and the human's decision on each (0011.03 Q15).

**After the blind review.** `reveal_labels` shows the judges beside the human's
blind verdict, so it runs only on what is saved. It never changes that verdict:
`human_pilot.jsonl` stays the blind record agreement is measured on, and a decision
goes to its own file with the blind value beside it.

**Checked, then stored.** `save` stores a decision only when it is one of the
item's values; otherwise it returns what is wrong and stores nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

from engine.repositories.dsld.runs import loader as runs_loader
from engine.repositories.dsld.runs import store as runs_store
from engine.repositories.dsld.runs.schema import ActualKind, Adjudication, Kind
from engine.services.extraction_eval.schema import RevealItem, RevealLabel, RevealOpinion, SheetCase


def reveal_labels(run_dir: str | Path, cases: list[SheetCase], judged: dict[str, dict[str, dict]]) -> list[RevealLabel]:
    """The labels in `cases` the human has saved, in that order, each with its reveal items.

    `judged` is each judge's stored verdict by `case_id`, by judge. A missed item
    is everything anyone listed on one piece of the label: quotes where one
    holds the other share an item. A claim item is a claim on which the human
    and the judges do not all give the same verdict. A label with no such item
    is still returned, with no items.
    """
    saved = runs_loader.read_human_verdicts(run_dir)
    out = []
    for c in cases:
        v = saved.get(c.case_id)
        if v is None:
            continue
        verdicts = {name: by_case[c.case_id] for name, by_case in judged.items()}

        # Missed: the human's quotes, then each judge's, joining the first item they overlap.
        groups = []
        listed = [("", m.quote, m.kind, m.comment, None) for m in v.missed]
        listed += [(name, m["quote"], m["kind"], "", m) for name, r in verdicts.items() for m in r["missed"]]
        for who, quote, kind, comment, m in listed:
            g = next((g for g in groups if any(quote in q or q in quote for q in g["all"])), None)
            if g is None:
                g = {"all": [], "blind": "none", "quotes": [], "comment": "", "opinions": {name: [] for name in judged}}
                groups.append(g)
            g["all"].append(quote)
            if not who:
                g["blind"], g["comment"] = kind, comment
                g["quotes"].append(quote)
            else:
                g["opinions"][who].append(RevealOpinion(kind, m["confidence"], m["rule"], quote, ""))
        items = [
            RevealItem("missed", " | ".join(sorted(set(g["all"]))), g["blind"], None, g["quotes"], g["comment"], g["opinions"])
            for g in groups
        ]

        # Claims: faithful and kind as one value, compared across the human and every judge.
        claims = {x.claim_id: x for x in c.claims}
        for x in v.claims:
            mine = "invented" if x.faithful == "invented" else f"{x.faithful}/{x.actual_kind}"
            opinions = {}
            for name, r in verdicts.items():
                y = next(y for y in r["claims"] if y["claim_id"] == x.claim_id)
                value = "invented" if y["faithful"] == "invented" else f"{y['faithful']}/{y['actual_kind']}"
                opinions[name] = [RevealOpinion(value, y["confidence"], y["rule"], y["evidence"], y["reason"])]
            if {o[0].value for o in opinions.values()} != {mine}:
                items.append(RevealItem("claim", x.claim_id, mine, claims[x.claim_id], [], x.comment, opinions))

        prompt_sha = next(iter(verdicts.values()))["prompt_sha"]
        out.append(RevealLabel(c.case_id, c.dsld_id, c.passage, prompt_sha, items))
    return out


def save(run_dir: str | Path, label: RevealLabel, item: str, final: str, gap: str) -> list[str]:
    """Check one decision on one item of `label`; store it if nothing is wrong. Returns what is wrong, `[]` once stored."""
    found = next((x for x in label.items if x.item == item), None)
    if found is None:
        return [f"{item!r} is not an item of {label.case_id}"]
    legal = (
        {*get_args(Kind), "none"}
        if found.item_kind == "missed"
        else {"invented", *(f"{f}/{k}" for f in ("yes", "meaning_changed") for k in get_args(ActualKind))}
    )
    problems = []
    if final not in legal:
        problems.append(f"{final!r} is not one of {sorted(legal)}")
    if not isinstance(gap, str):
        problems.append("the note must be text")
    if problems:
        return problems
    runs_store.save_adjudication(
        run_dir,
        Adjudication(
            case_id=label.case_id,
            dsld_id=label.dsld_id,
            item_kind=found.item_kind,
            item=item,
            blind=found.blind,
            final=final,
            gap=gap,
            prompt_sha=label.prompt_sha,
            decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    return []
