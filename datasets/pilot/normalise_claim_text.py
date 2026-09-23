"""Normalise an entry `text` the way gold-1000 does.

Gold lowercases claim text -- including Title Case marketing lines -- but
keeps genuine acronyms (GI, EPA, DHA, TPP, AGEs, TM/R markers). So this
lowercases a word only when it is not an acronym, rather than calling
.lower() on the whole string, which would destroy those 12 gold-style cases.
"""
import re

# A word kept as-is: all-caps (GI, EPA, ATP), or caps+digits/suffix (AGEs,
# BC30, DEEP2, GHI). Mixed-case brand tokens (optiCleanse) are also kept.
ACRONYM = re.compile(r"^[A-Z0-9]{2,}[a-z]?$|^[A-Z]+\d[A-Za-z0-9]*$")


def is_acronym(word: str) -> bool:
    core = word.strip(".,:;()[[]{}\"'").replace("(TM)", "").replace("(R)", "")
    core = core.strip(".,:;()[]{}\"'")
    if not core or not any(c.isalpha() for c in core):
        return False
    if ACRONYM.match(core):
        return True
    # mixed case with an interior capital, e.g. optiCleanse, pH
    return any(c.isupper() for c in core[1:])


def normalise(text: str) -> str:
    t = text.strip().lstrip("-–—•*").strip()
    t = re.sub(r"\*+$", "", t).strip()
    return " ".join(w if is_acronym(w) else w.lower() for w in t.split())
