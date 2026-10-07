"""Mine boilerplate candidates: the most frequent normalised sentences in the DSLD dump.

Produced docs/boilerplate-candidates.md (see its "How these were found"). Run from
the repo root:

    .venv/bin/python docs/boilerplate-candidates.py OUT.md [N_FILES]

Refuses to overwrite OUT.md, so ticks already made in the checklist are never lost.
N_FILES limits the run to the first N files, for a quick check.

The normaliser here is a sketch of spec 0005's, written before that spec; it has no
age-limit exception and a shorter unit list.
"""

import collections
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

from engine.services.dsld_cleaning.service import clean

DUMP = "datasets/DSLD-full-database-JSON/*/*.json"
TOP = 400
MIN_CHARS = 15

UNITS = r"(?:mg|mcg|µg|ug|g|kg|iu|ml|l|oz|fl oz|lb|lbs|cal|kcal|capsules?|caps|tablets?|softgels?|servings?|gummies|gummy|drops?|scoops?|packets?|billion|million|cfu)"
QTY = re.compile(r"(?<![a-z0-9-])\d[\d,]*(?:\.\d+)?\s*(?:%|" + UNITS + r"\b)?", re.I)
SENT = re.compile(r"(?<=[.!?;])\s+|\n")


def norm(s):
    s = s.lower()
    s = QTY.sub(" ", s)
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def work(paths):
    labels, occurrences, sample = collections.Counter(), collections.Counter(), {}
    for p in paths:
        try:
            rec = json.load(open(p))
        except Exception:
            continue
        c = clean(rec)
        seen = set()
        for s in c.segments:
            if s.paths == ["fullName"]:
                continue
            for t in SENT.split(s.text):
                n = norm(t)
                if len(n) >= MIN_CHARS:
                    seen.add(n)
                    occurrences[n] += 1
        for n in seen:
            labels[n] += 1
            sample.setdefault(n, c.dsld_id)
    return labels, occurrences, sample


if __name__ == "__main__":
    out = sys.argv[1]
    if os.path.exists(out):
        sys.exit(f"{out} exists; refusing to overwrite (it may hold ticks)")
    files = sorted(glob.glob(DUMP))
    if len(sys.argv) > 2:
        files = files[: int(sys.argv[2])]
    chunks = [files[i::48] for i in range(48)]
    L, O, sample = collections.Counter(), collections.Counter(), {}
    with Pool(10) as pool:
        # Chunks finish in any order, so the sample label differs between runs.
        for l, o, sm in pool.imap_unordered(work, chunks):
            L.update(l)
            O.update(o)
            for k, v in sm.items():
                sample.setdefault(k, v)
    with open(out, "w") as f:
        f.write("# Boilerplate candidates\n\n")
        for n, _ in O.most_common(TOP):
            f.write(f"- [ ] {O[n]:,} times · {L[n]:,} labels · {n} · "
                    f"[{sample[n]}](https://dsld.od.nih.gov/label/{sample[n]})\n")
    print(len(files), "files;", len(L), "distinct sentences")
