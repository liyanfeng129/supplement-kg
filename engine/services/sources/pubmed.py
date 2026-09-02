"""PubMed abstract retrieval via NCBI E-utilities.

Fetches candidate abstracts for supplement/health-claim studies, to be fed
into downstream PICO extraction and ontology matching.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET

import requests

from engine import config

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Study designs that can support a PICO frame. Anything outside this is a poor
# candidate for extraction: no intervention, no comparator, or no primary data.
INTERVENTIONAL_PT = [
    "Randomized Controlled Trial[pt]",
    "Controlled Clinical Trial[pt]",
    "Clinical Trial[pt]",
    "Meta-Analysis[pt]",
    "Systematic Review[pt]",
]

# Excluded regardless of other filters: no extractable human PICO.
EXCLUDED_PT = [
    "Editorial[pt]",
    "Comment[pt]",
    "Letter[pt]",
    "News[pt]",
    "Retracted Publication[pt]",
]

HUMAN_FILTER = "Humans[mh]"


@dataclass
class QuerySpec:
    """Declarative description of one PubMed search.

    A spec is anchored to a claim: `substance` is the supplement/nutrient and
    `health_terms` are the outcome concepts. Keeping them separate (rather than
    one opaque query string) lets downstream extraction check whether the PICO
    it found actually matches the claim we searched for.
    """

    substance: str
    health_terms: list[str] = field(default_factory=list)
    substance_synonyms: list[str] = field(default_factory=list)
    claim_id: str | None = None
    claim_text: str | None = None

    # Filters
    interventional_only: bool = True
    humans_only: bool = True
    exclude_non_research: bool = True
    require_abstract: bool = True
    english_only: bool = True
    date_from: str | None = None   # "YYYY/MM/DD"
    date_to: str | None = None

    def _or_block(self, terms: Iterable[str], field_tag: str = "tiab") -> str:
        parts = []
        for t in terms:
            t = t.strip()
            if not t:
                continue
            # Quote multi-word terms; search both MeSH and title/abstract.
            quoted = f'"{t}"' if " " in t else t
            parts.append(f"{quoted}[{field_tag}]")
            parts.append(f"{quoted}[mh]")
        return "(" + " OR ".join(parts) + ")" if parts else ""

    def build(self) -> str:
        """Render this spec as a PubMed query string."""
        clauses: list[str] = []

        subst_terms = [self.substance, *self.substance_synonyms]
        subst = self._or_block(subst_terms)
        if subst:
            clauses.append(subst)

        if self.health_terms:
            clauses.append(self._or_block(self.health_terms))

        if self.interventional_only:
            clauses.append("(" + " OR ".join(INTERVENTIONAL_PT) + ")")
        if self.humans_only:
            clauses.append(HUMAN_FILTER)
        if self.exclude_non_research:
            for pt in EXCLUDED_PT:
                clauses.append(f"NOT {pt}")
        if self.require_abstract:
            clauses.append("hasabstract")
        if self.english_only:
            clauses.append("English[lang]")

        query = " AND ".join(c for c in clauses if not c.startswith("NOT"))
        for c in clauses:
            if c.startswith("NOT"):
                query += f" {c}"

        if self.date_from or self.date_to:
            lo = self.date_from or "1800/01/01"
            hi = self.date_to or "3000/01/01"
            query += f' AND ("{lo}"[dp] : "{hi}"[dp])'

        return query


@dataclass
class Abstract:
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    year: str | None = None
    doi: str | None = None
    publication_types: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    # Provenance: which claim/query surfaced this record.
    claim_id: str | None = None
    query: str | None = None


class PubMedClient:
    """Thin E-utilities client with rate limiting and history-server paging."""

    def __init__(
        self,
        api_key: str | None = None,
        email: str | None = None,
        tool: str = "supplement-kg",
        timeout: int = 30,
        max_retries: int = 4,
    ):
        self.api_key = api_key or config.NCBI_API_KEY
        self.email = email or config.NCBI_EMAIL
        self.tool = tool
        self.timeout = timeout
        # NCBI allows 3 req/s without a key, 10 with one. The published rates
        # are enforced with little slack, so leave headroom rather than riding
        # the limit -- a 429 mid-run costs more than the extra latency.
        self._min_interval = 0.15 if self.api_key else 0.4
        self._last_call = 0.0
        self._max_retries = max_retries
        self._session = requests.Session()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _params(self, **kw) -> dict:
        p = {"tool": self.tool, **kw}
        if self.api_key:
            p["api_key"] = self.api_key
        if self.email:
            p["email"] = self.email
        return p

    def _get(self, endpoint: str, **params):
        """GET with backoff. NCBI answers overload with 429/5xx, and a long
        multi-claim run will hit them; retrying is the difference between a
        partial corpus and a complete one."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                r = self._session.get(
                    f"{EUTILS}/{endpoint}",
                    params=self._params(**params),
                    timeout=self.timeout,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}", response=r)
                r.raise_for_status()
                return r
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt == self._max_retries - 1:
                    break
                time.sleep(2.0 ** attempt)
        raise RuntimeError(f"E-utilities request failed after {self._max_retries} attempts") from last_exc

    def search(self, query: str, retmax: int = 0) -> dict:
        """Run esearch with history enabled. Returns count + history handles."""
        r = self._get(
            "esearch.fcgi",
            db="pubmed",
            term=query,
            usehistory="y",
            retmode="json",
            retmax=retmax,
        )
        result = r.json()["esearchresult"]
        return {
            "count": int(result.get("count", 0)),
            "webenv": result.get("webenv"),
            "query_key": result.get("querykey"),
            "pmids": result.get("idlist", []),
            "translation": result.get("querytranslation"),
        }

    def fetch_records(
        self,
        webenv: str,
        query_key: str,
        total: int,
        limit: int | None = None,
        batch_size: int = 200,
    ) -> Iterator[ET.Element]:
        """Yield PubmedArticle XML elements from the history server."""
        want = total if limit is None else min(limit, total)
        for start in range(0, want, batch_size):
            n = min(batch_size, want - start)
            r = self._get(
                "efetch.fcgi",
                db="pubmed",
                WebEnv=webenv,
                query_key=query_key,
                retstart=start,
                retmax=n,
                retmode="xml",
            )
            root = ET.fromstring(r.content)
            for art in root.findall(".//PubmedArticle"):
                yield art


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def parse_article(art: ET.Element) -> Abstract:
    pmid = _text(art.find(".//PMID"))
    title = _text(art.find(".//ArticleTitle"))

    # Structured abstracts have multiple labelled sections; keep the labels,
    # they carry PICO signal (OBJECTIVE / METHODS / RESULTS / CONCLUSIONS).
    chunks = []
    for node in art.findall(".//Abstract/AbstractText"):
        label = node.get("Label")
        body = _text(node)
        if not body:
            continue
        chunks.append(f"{label}: {body}" if label else body)
    abstract = "\n".join(chunks)

    journal = _text(art.find(".//Journal/Title")) or None
    year = _text(art.find(".//JournalIssue/PubDate/Year")) or None
    if not year:
        medline = _text(art.find(".//JournalIssue/PubDate/MedlineDate"))
        year = medline[:4] if medline else None

    doi = None
    for aid in art.findall(".//ArticleId"):
        if aid.get("IdType") == "doi":
            doi = _text(aid)
            break

    pts = [_text(p) for p in art.findall(".//PublicationType")]
    mesh = [_text(d) for d in art.findall(".//MeshHeading/DescriptorName")]

    return Abstract(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        doi=doi,
        publication_types=[p for p in pts if p],
        mesh_terms=[m for m in mesh if m],
    )


def run_query(
    spec: QuerySpec,
    client: PubMedClient | None = None,
    limit: int | None = 50,
    batch_size: int = 200,
) -> tuple[list[Abstract], dict]:
    """Execute one QuerySpec. Returns (abstracts, run manifest).

    `limit` caps how many records are downloaded; the manifest still records
    the full result count, so you can see what you truncated.
    """
    client = client or PubMedClient()
    query = spec.build()
    started = datetime.now(timezone.utc).isoformat()

    res = client.search(query)
    abstracts: list[Abstract] = []

    if res["count"] and res["webenv"]:
        for art in client.fetch_records(
            res["webenv"], res["query_key"], res["count"],
            limit=limit, batch_size=batch_size,
        ):
            a = parse_article(art)
            a.claim_id = spec.claim_id
            a.query = query
            abstracts.append(a)

    pmids = sorted(a.pmid for a in abstracts)
    manifest = {
        "claim_id": spec.claim_id,
        "claim_text": spec.claim_text,
        "substance": spec.substance,
        "health_terms": spec.health_terms,
        "query": query,
        "query_translation": res.get("translation"),
        "total_hits": res["count"],
        "fetched": len(abstracts),
        "limit": limit,
        "executed_at": started,
        "pmids": pmids,
        "pmid_hash": hashlib.sha256(",".join(pmids).encode()).hexdigest()[:16],
        "spec": asdict(spec),
    }
    return abstracts, manifest


def save_run(
    abstracts: list[Abstract],
    manifest: dict,
    out_dir: str | Path,
    run_name: str | None = None,
) -> dict[str, Path]:
    """Write payload (gitignored) and manifest (committable) side by side."""
    out_dir = Path(out_dir)
    name = run_name or (manifest.get("claim_id") or "run")
    raw_dir = out_dir / "raw"
    man_dir = out_dir / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{name}.jsonl"
    with raw_path.open("w") as fh:
        for a in abstracts:
            fh.write(json.dumps(asdict(a)) + "\n")

    man_path = man_dir / f"{name}.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    return {"raw": raw_path, "manifest": man_path}
