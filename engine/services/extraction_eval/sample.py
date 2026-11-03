"""Draw the evaluation's labels from a run's profile: the pilot, the large set, the human set.

Seeded and stratified, no model and no judge (spec 0011.02). A label's
**rarest type** is over all its claims, both kinds together, first match
wins: `rare` if any claim is `spread` or `invented`, else `stitched` if any
is, else `verbatim`; a label without claims has none. Its **stratum** is
`(group, rarest)`, one of the 10 in `STRATA`.

Within a stratum labels are ordered by `sha256(f"{seed}:{dsld_id}")`, and
every set is a prefix of that order: the pilot is its head, the large set
the next `large`, and each set's human part the head of that set. A label's
place depends only on the seed and its own id, never on the run's row order.
A quota above what is left takes all of it.

`draw_pilot` and `draw` are pure; `build_pilot` and `build_evaluation_set`
read the stored profile, draw, and save beside the run. They never profile
again.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from engine.repositories.dsld.runs import loader as runs_loader
from engine.repositories.dsld.runs import store as runs_store
from engine.repositories.dsld.runs.schema import STRATA, SampledLabel, Stratum, StratumSize
from engine.services.extraction_eval import profile as profile_service
from engine.services.extraction_eval.schema import Profile, Quota, Sample


def draw_pilot(profile: Profile, quotas: dict[Stratum, Quota], seed: int) -> list[SampledLabel]:
    """The pilot: the head `pilot` of each stratum's seeded order, its first `pilot_human` flagged.

    Pure. Returned in stratum order, then rank. Raises on quotas that do not
    name exactly the 10 strata, or that do not nest.
    """
    # Quotas: exactly the 10 strata, each nesting.
    if set(quotas) != set(STRATA):
        raise ValueError(
            f"quotas must name exactly the 10 strata; missing {sorted(set(STRATA) - set(quotas), key=str)}, "
            f"unknown {sorted(set(quotas) - set(STRATA), key=str)}"
        )
    for s, q in quotas.items():
        if q.pilot_human > q.pilot or q.human > q.large:
            raise ValueError(f"quota for {s} does not nest: {q}")

    # Stratum per label, by its rarest type over both kinds.
    by_stratum: dict[Stratum, list[int]] = {s: [] for s in STRATA}
    for label in profile.labels:
        types = set(label.purpose_types) | set(label.risk_types)
        if not types:
            rarest = None
        elif types & {"spread", "invented"}:
            rarest = "rare"
        elif "stitched" in types:
            rarest = "stitched"
        else:
            rarest = "verbatim"
        by_stratum[(label.group, rarest)].append(label.dsld_id)

    # The head of each stratum's seeded order.
    pilot: list[SampledLabel] = []
    for s in STRATA:
        order = sorted(by_stratum[s], key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
        q = quotas[s]
        for rank, dsld_id in enumerate(order[: q.pilot]):
            pilot.append(SampledLabel(f"label:{dsld_id}", dsld_id, s, rank, rank < q.pilot_human))
    return pilot


def draw(
    profile: Profile, quotas: dict[Stratum, Quota], seed: int, pilot: list[SampledLabel]
) -> Sample:
    """The large set after the pilot: the next `large` of each stratum's order, its first `human` flagged.

    Pure. `pilot` must be exactly `draw_pilot(profile, quotas, seed)`; a
    mismatch means the profile, quotas or seed changed since the pilot was
    drawn, and raises. Raises too on a non-empty stratum the pilot used up,
    whose weight `in_run / large` would be undefined.
    """
    # The pilot must be the one these inputs draw; draw_pilot checks the quotas.
    if pilot != draw_pilot(profile, quotas, seed):
        raise ValueError(
            "the pilot is not the one this profile, these quotas and this seed draw: "
            "one of them changed since the pilot was drawn"
        )

    # Stratum per label, by its rarest type over both kinds.
    by_stratum: dict[Stratum, list[int]] = {s: [] for s in STRATA}
    for label in profile.labels:
        types = set(label.purpose_types) | set(label.risk_types)
        if not types:
            rarest = None
        elif types & {"spread", "invented"}:
            rarest = "rare"
        elif "stitched" in types:
            rarest = "stitched"
        else:
            rarest = "verbatim"
        by_stratum[(label.group, rarest)].append(label.dsld_id)

    # After the pilot in each stratum's seeded order: the large set, its head the human set.
    large: list[SampledLabel] = []
    sizes: list[StratumSize] = []
    for s in STRATA:
        order = sorted(by_stratum[s], key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
        q = quotas[s]
        head = order[: q.pilot]
        rest = order[len(head) : len(head) + q.large]
        if order and not rest:
            raise ValueError(
                f"stratum {s} has {len(order)} labels, the pilot takes {len(head)} and the large "
                f"quota is {q.large}: the large set gets none, so its weight is undefined. Lower "
                f"its pilot quota or raise its large one."
            )
        for j, dsld_id in enumerate(rest):
            large.append(SampledLabel(f"label:{dsld_id}", dsld_id, s, len(head) + j, j < q.human))
        sizes.append(
            StratumSize(
                stratum=s,
                in_run=len(order),
                pilot=len(head),
                pilot_human=min(q.pilot_human, len(head)),
                large=len(rest),
                human=min(q.human, len(rest)),
            )
        )
    return Sample(seed, pilot, large, sizes)


def build_pilot(run_dir: str | Path, quotas: dict[Stratum, Quota], seed: int) -> list[SampledLabel]:
    """Draw a run's pilot from its stored profile and save it in the run's `eva/`; return it.

    Raises if the run has no profile, a stale one or one of another
    `classifier_version`, and if a pilot is already stored. The manifest
    records the seed, the quotas, the pilot's sizes per stratum and the count
    of failed labels, which no stratum holds.
    """
    # The stored profile, current and of this classifier; never profile again.
    manifest, labels = runs_loader.read_profile(run_dir)
    if manifest.get("classifier_version") != profile_service.VERSION:
        raise ValueError(
            f"profile of run {run_dir} has classifier_version {manifest.get('classifier_version')}, "
            f"not {profile_service.VERSION}: delete it and profile the run again"
        )
    prof = Profile(
        labels,
        {(g, k, t): n for g, ks in manifest["claim_counts"].items() for k, ts in ks.items() for t, n in ts.items()},
        manifest["label_counts"],
        manifest["failed_ids"],
    )

    pilot = draw_pilot(prof, quotas, seed)

    runs_store.save_pilot(
        run_dir,
        pilot,
        {
            "run": Path(run_dir).name,
            "classifier_version": profile_service.VERSION,
            "seed": seed,
            "failed": len(prof.failed_ids),
            "quotas": [{"group": s[0], "rarest": s[1], **asdict(quotas[s])} for s in STRATA],
            "sizes": [
                {
                    "group": s[0],
                    "rarest": s[1],
                    "pilot": sum(l.stratum == s for l in pilot),
                    "pilot_human": sum(l.stratum == s and l.in_human for l in pilot),
                }
                for s in STRATA
            ],
        },
    )
    return pilot


def build_evaluation_set(run_dir: str | Path, quotas: dict[Stratum, Quota], seed: int) -> Sample:
    """Draw a run's large and human sets after its stored pilot, and save them in its `eva/`.

    Raises as `build_pilot` does, if no pilot is stored, if the stored pilot
    is not the one these quotas and this seed draw, and if an evaluation set
    is already stored. The manifest records the seed, the quotas and every
    stratum's sizes, from which the weights `in_run / large` and
    `in_run / human` are computed where used.
    """
    # The stored profile, current and of this classifier; never profile again.
    manifest, labels = runs_loader.read_profile(run_dir)
    if manifest.get("classifier_version") != profile_service.VERSION:
        raise ValueError(
            f"profile of run {run_dir} has classifier_version {manifest.get('classifier_version')}, "
            f"not {profile_service.VERSION}: delete it and profile the run again"
        )
    prof = Profile(
        labels,
        {(g, k, t): n for g, ks in manifest["claim_counts"].items() for k, ts in ks.items() for t, n in ts.items()},
        manifest["label_counts"],
        manifest["failed_ids"],
    )

    pilot = [c.label for c in runs_loader.read_pilot(run_dir)]
    sample = draw(prof, quotas, seed, pilot)

    runs_store.save_evaluation_set(
        run_dir,
        sample.large,
        {
            "run": Path(run_dir).name,
            "classifier_version": profile_service.VERSION,
            "seed": seed,
            "failed": len(prof.failed_ids),
            "quotas": [{"group": s[0], "rarest": s[1], **asdict(quotas[s])} for s in STRATA],
            "sizes": [
                {"group": z.stratum[0], "rarest": z.stratum[1], **{k: v for k, v in asdict(z).items() if k != "stratum"}}
                for z in sample.sizes
            ],
        },
    )
    return sample
