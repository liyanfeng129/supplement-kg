"""DSLD label retrieval via the NIH ODS Label API.

Fetches product label records -- ingredients, claims, statements -- for the
DSLD IDs carried by iDISK DSP rows, to be joined against the claim/PICO side
of the graph.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import requests

# The public API host. Note this is *not* dsld.od.nih.gov -- that host sits
# behind a Cloudflare interstitial and answers programmatic GETs with a 403.
DSLD_API = "https://api.ods.od.nih.gov/dsld/v9"

# Elasticsearch's default max_result_window. `search-filter` honours `size` up
# to this, but a `from` at or past it yields an empty `hits` with HTTP 200 --
# another of this API's silent failures. Any bulk slice must stay under it.
MAX_RESULT_WINDOW = 10_000
PAGE_SIZE = 1_000

# Product type LanguaL codes, used only to subdivide a (year, status) slice
# that is still over the window. Values from the v9 `search-filter` docs.
PRODUCT_TYPE_CODES = (
    "a1305", "a1306", "a1326", "a1310", "a1302", "a1299",
    "a1316", "a1315", "a1317", "a1309", "a1325",
)

# DSLD's first entry year; earlier years return nothing.
FIRST_ENTRY_YEAR = 2011

# The API guide documents 1,000 requests/hour/IP -- 0.28 req/s, or one request
# every 3.6s. No response carries a rate-limit header, so this number cannot be
# discovered at runtime; it has to be encoded here.
HOURLY_REQUEST_BUDGET = 1000
# 5% headroom. The exact interval would sit right on the limit, and a run
# holds no margin for the server counting differently (a fixed window, or
# retries billed against the same budget) -- being throttled costs far more
# than the ~3 minutes the margin adds per thousand requests.
SAFE_REQUEST_INTERVAL = 3600 / (HOURLY_REQUEST_BUDGET * 0.95)


@dataclass
class Ingredient:
    """One ingredient row, flattened to the fields the KG actually consumes."""

    name: str
    ingredient_id: int | None = None
    category: str | None = None
    ingredient_group: str | None = None
    unii_code: str | None = None
    quantity: float | None = None
    unit: str | None = None
    # Percent of daily value, when the label declares one.
    daily_value_percent: float | None = None
    alternate_names: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    # Sub-rows of a proprietary blend, kept nested so the blend structure
    # survives -- a flat list would lose which ingredients belong together.
    nested: list["Ingredient"] = field(default_factory=list)


@dataclass
class Label:
    dsld_id: int
    full_name: str
    brand_name: str | None = None
    upc_sku: str | None = None
    product_type: str | None = None
    physical_state: str | None = None
    serving_sizes: list[str] = field(default_factory=list)
    servings_per_container: str | None = None
    target_groups: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    # Label statements keyed by type ("Precautions re: All Other", ...).
    statements: list[dict] = field(default_factory=list)
    ingredients: list[Ingredient] = field(default_factory=list)
    other_ingredients: list[str] = field(default_factory=list)
    off_market: bool | None = None
    entry_date: str | None = None
    # Provenance: which iDISK product row asked for this label.
    idisk_id: str | None = None


class DSLDNotFound(LookupError):
    """No label exists for the requested ID.

    The API answers an unknown ID with HTTP 200 and an empty list rather than
    a 404, so callers cannot rely on status codes to detect this.
    """


class _RateGate:
    """Shared, self-adjusting pace for concurrent DSLD requests.

    DSLD's documented limit is 1,000 requests/hour/IP -- a budget over time, so
    no worker count can respect it on its own: a burst well under any
    concurrency ceiling still drains the hour and gets everything afterwards
    refused. This gate paces all workers against one shared interval, which
    defaults to the documented 3.6s and only ever grows -- doubling on each
    429/403 and decaying back toward `floor` on sustained success.

    The floor is therefore the real throttle, not `workers`. Threads only
    overlap the ~200ms of latency inside each 3.6s slot; they do not and must
    not raise the request rate.
    """

    def __init__(self, floor: float = SAFE_REQUEST_INTERVAL, ceiling: float = 60.0):
        self._floor = floor
        self._ceiling = ceiling
        self._interval = floor
        self._next = 0.0
        self._ok_streak = 0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until this caller's turn, keeping the global pace."""
        with self._lock:
            now = time.monotonic()
            # Serialise the *slot*, not the sleep: holding the lock while
            # sleeping would collapse the pool to one worker.
            start = max(now, self._next)
            self._next = start + self._interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalise(self) -> None:
        """Back off after a refusal."""
        with self._lock:
            self._ok_streak = 0
            self._interval = min(max(self._interval * 2, self._floor * 2), self._ceiling)

    def reward(self) -> None:
        """Ease back up after a run of clean responses."""
        with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 50 and self._interval > self._floor:
                self._interval = max(self._interval * 0.9, self._floor)
                self._ok_streak = 0


class DSLDClient:
    """Thin DSLD Label API client with rate limiting and retries."""

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 4,
        min_interval: float = 0.2,
    ):
        self.timeout = timeout
        # ODS publishes no rate limit, so this is self-imposed politeness --
        # a full DSP sweep is ~90k requests and should not look like a flood.
        self._min_interval = min_interval
        self._last_call = 0.0
        self._max_retries = max_retries
        self._session = requests.Session()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _get(self, path: str, **params):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                r = self._session.get(
                    f"{DSLD_API}/{path}",
                    params=params or None,
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
        raise RuntimeError(
            f"DSLD request failed after {self._max_retries} attempts: {path}"
        ) from last_exc

    def fetch_raw(self, dsld_id: int | str) -> dict:
        """Return the raw label JSON for one DSLD ID."""
        r = self._get(f"label/{dsld_id}")
        doc = r.json()
        # Unknown IDs come back as [] with a 200; a hit is a JSON object.
        if not doc or not isinstance(doc, dict):
            raise DSLDNotFound(str(dsld_id))
        return doc

    def fetch(self, dsld_id: int | str, idisk_id: str | None = None) -> Label:
        """Return one parsed label."""
        label = parse_label(self.fetch_raw(dsld_id))
        label.idisk_id = idisk_id
        return label

    def fetch_concurrent(
        self,
        dsld_ids: Iterable[int | str],
        workers: int = 4,
        skip_missing: bool = True,
        on_failure: "list | None" = None,
    ) -> Iterator[Label]:
        """Yield parsed labels for many IDs, paced to DSLD's hourly budget.

        **The API guide documents 1,000 requests/hour/IP** -- 0.28 req/s, one
        request every 3.6s. That, not concurrency, is the binding constraint,
        and `_RateGate` enforces it: `workers` only overlaps the ~200ms of
        latency within each slot, so raising it does not raise throughput.

        Measured behaviour matches the documented budget. Stepping 1 -> 64
        workers (2026-09-05) looked clean up to 8 workers at ~34 req/s -- 122x
        the limit -- purely because a short burst spends the hour rather than
        being refused up front. Past that the endpoint returned 429 at *every*
        concurrency including 1, then escalated to a bare 403 from the load
        balancer that took minutes to decay. So a benchmark that "passes"
        proves nothing here; it is just spending the budget faster.

        At the documented rate a full per-ID sweep is days, not hours: ~58h
        for the 57,837 labels new since iDISK. Prefer `dsld_bulk.harvest()`
        (~371 requests) for anything the summary documents can answer, and
        reserve this for fields only `label/{id}` carries -- chiefly the
        statement text purpose/risk extraction needs.

        IDs that exhaust their retries are appended to `on_failure` (when
        given) rather than aborting a run that may already be hours old.

        Results arrive out of order, and are yielded as they complete so a
        long sweep can be checkpointed rather than held in memory.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        local = threading.local()
        gate = _RateGate()

        def one(did: int | str) -> Label | None:
            session = getattr(local, "session", None)
            if session is None:
                session = local.session = requests.Session()
            for attempt in range(self._max_retries):
                gate.wait()
                try:
                    r = session.get(f"{DSLD_API}/label/{did}", timeout=self.timeout)
                    # 403 shows up here too: once the budget is blown the load
                    # balancer rejects outright, and it is recoverable, so it
                    # has to back off rather than raise like a real 403.
                    if r.status_code in (403, 429):
                        gate.penalise()
                        raise requests.HTTPError(str(r.status_code), response=r)
                    if r.status_code in (500, 502, 503, 504):
                        raise requests.HTTPError(str(r.status_code), response=r)
                    r.raise_for_status()
                    gate.reward()
                    doc = r.json()
                    if not doc or not isinstance(doc, dict):
                        if skip_missing:
                            return None
                        raise DSLDNotFound(str(did))
                    return parse_label(doc)
                except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
                    if attempt == self._max_retries - 1:
                        if on_failure is None:
                            raise
                        on_failure.append(did)
                        return None
                    time.sleep(2.0 ** attempt)
            return None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, d) for d in dsld_ids]
            for fut in as_completed(futures):
                label = fut.result()
                if label is not None:
                    yield label

    def fetch_many(
        self,
        dsld_ids: Iterable[int | str],
        idisk_id: str | None = None,
        skip_missing: bool = True,
    ) -> Iterator[Label]:
        """Yield parsed labels for many IDs.

        With `skip_missing` (the default) an ID absent from DSLD is passed
        over rather than aborting the sweep -- iDISK carries stale IDs, and a
        single retired product should not sink a 90k-row run.
        """
        for did in dsld_ids:
            try:
                yield self.fetch(did, idisk_id=idisk_id)
            except DSLDNotFound:
                if not skip_missing:
                    raise


    def search_count(self, **filters) -> int:
        """Number of labels matching `filters`, without fetching them.

        `search-filter` reports the match count in `stats.count` even when
        `size=1`, which makes it cheap to size a slice before paging it.
        """
        r = self._get("search-filter", q="*", size=1, **filters)
        return r.json().get("stats", {}).get("count", 0)

    def search_page(self, frm: int, size: int, **filters) -> list[dict]:
        """One page of raw label documents from `search-filter`."""
        r = self._get(
            "search-filter",
            q="*",
            sort_by="fullName.keyword",
            sort_order="asc",
            **{"from": frm, "size": size},
            **filters,
        )
        return [h["_source"] | {"id": int(h["_id"])} for h in r.json().get("hits", [])]

    def plan_slices(self) -> list[dict]:
        """Partition the whole database into slices that fit the result window.

        Paging alone cannot reach past label 10,000, so the corpus is cut by
        entry year, then by market status, then -- only where still needed --
        by product type. Each key is exhaustive and disjoint, so the slices
        together cover every label exactly once; the caller can assert that by
        comparing the summed counts against an unfiltered `search_count()`.
        """
        this_year = datetime.now(timezone.utc).year
        slices: list[dict] = []
        for year in range(FIRST_ENTRY_YEAR, this_year + 1):
            for status in (0, 1):
                base = {"date_start": year, "date_end": year, "status": status}
                count = self.search_count(**base)
                if count == 0:
                    continue
                if count < MAX_RESULT_WINDOW:
                    slices.append({**base, "expected": count})
                    continue
                for code in PRODUCT_TYPE_CODES:
                    sub = {**base, "product_type": code}
                    sub_count = self.search_count(**sub)
                    if sub_count:
                        slices.append({**sub, "expected": sub_count})
        return slices

    def iter_all(self, slices: list[dict] | None = None) -> Iterator[dict]:
        """Yield every raw label document in DSLD, slice by slice.

        Deduplication is the caller's job only across slices that a future
        DSLD schema change might overlap; the current keys are disjoint.
        """
        for sl in slices if slices is not None else self.plan_slices():
            filters = {k: v for k, v in sl.items() if k != "expected"}
            for frm in range(0, sl["expected"], PAGE_SIZE):
                page = self.search_page(frm, PAGE_SIZE, **filters)
                if not page:
                    break
                yield from page


def parse_dsld_ids(cell: object) -> list[str]:
    """Split an iDISK DSP `DSLD_ID` cell into individual IDs.

    The column is pipe-delimited and often has a leading empty field, e.g.
    `" | 34066 | 73755"`, so a naive `int(cell)` fails on most rows.
    """
    if cell is None:
        return []
    text = str(cell)
    if not text or text.lower() == "nan":
        return []
    return [tok for tok in re.split(r"\s*\|\s*", text.strip()) if tok.strip().isdigit()]


def _parse_ingredient(row: dict) -> Ingredient:
    # A row carries one quantity entry per serving size; take the first, which
    # is the label's primary declared serving.
    qty = (row.get("quantity") or [{}])[0]
    dv = (qty.get("dailyValueTargetGroup") or [{}])[0]
    return Ingredient(
        name=row.get("name") or "",
        ingredient_id=row.get("ingredientId"),
        category=row.get("category") or None,
        ingredient_group=row.get("ingredientGroup") or None,
        unii_code=row.get("uniiCode") or None,
        quantity=qty.get("quantity"),
        unit=qty.get("unit") or None,
        daily_value_percent=dv.get("percent"),
        alternate_names=[a for a in (row.get("alternateNames") or []) if a],
        forms=[f for f in (row.get("forms") or []) if f],
        nested=[_parse_ingredient(n) for n in (row.get("nestedRows") or [])],
    )


def _names(items: object, key: str = "name") -> list[str]:
    """Pull `key` out of a list of small objects, tolerating plain strings.

    Several DSLD fields (`targetGroups`) are plain string lists while others
    (`otheringredients.ingredients`) are objects, so both shapes are accepted.
    """
    out = []
    for it in items or []:
        val = it.get(key) if isinstance(it, dict) else it
        if val:
            out.append(val)
    return out


def _langual(node: object) -> str | None:
    """Readable value of a LanguaL-coded field.

    DSLD encodes product type, physical state and claim type as LanguaL codes
    with a description alongside ({"langualCode": "E0159",
    "langualCodeDescription": "Capsule"}). The description is what a reader
    or a matcher wants; the bare code is meaningless without the thesaurus.
    """
    if isinstance(node, dict):
        return node.get("langualCodeDescription") or node.get("langualCode") or None
    return node or None


def _serving_size(s: dict) -> str:
    """Render one serving size as "2 Capsule(s)", or "1-2 Capsule(s)"."""
    lo, hi = s.get("minQuantity"), s.get("maxQuantity")
    if lo is None and hi is None:
        qty = ""
    elif hi is None or lo == hi:
        qty = str(lo if lo is not None else hi)
    else:
        qty = f"{lo}-{hi}"
    return " ".join(p for p in (qty, s.get("unit")) if p).strip()


def parse_label(doc: dict) -> Label:
    """Flatten one raw DSLD label document into a `Label`."""
    serving_sizes = [_serving_size(s) for s in (doc.get("servingSizes") or [])]
    other = doc.get("otheringredients") or {}

    return Label(
        dsld_id=doc["id"],
        full_name=doc.get("fullName") or "",
        brand_name=doc.get("brandName") or None,
        upc_sku=doc.get("upcSku") or None,
        product_type=_langual(doc.get("productType")),
        physical_state=_langual(doc.get("physicalState")),
        serving_sizes=[s for s in serving_sizes if s],
        servings_per_container=doc.get("servingsPerContainer") or None,
        target_groups=_names(doc.get("targetGroups")),
        claims=[c for c in (_langual(c) for c in (doc.get("claims") or [])) if c],
        statements=[
            {"type": s.get("type"), "notes": s.get("notes")}
            for s in (doc.get("statements") or [])
        ],
        ingredients=[_parse_ingredient(r) for r in (doc.get("ingredientRows") or [])],
        other_ingredients=_names(other.get("ingredients")),
        off_market=bool(doc["offMarket"]) if doc.get("offMarket") is not None else None,
        entry_date=doc.get("entryDate") or None,
    )


def fetch_products(
    rows: Iterable[tuple[str, object]],
    client: DSLDClient | None = None,
    limit: int | None = None,
) -> tuple[list[Label], dict]:
    """Fetch labels for `(idisk_id, dsld_id_cell)` pairs.

    Returns (labels, manifest). The manifest records what was requested versus
    what came back, so a run's misses are visible without re-querying.
    """
    client = client or DSLDClient()
    started = datetime.now(timezone.utc).isoformat()

    labels: list[Label] = []
    missing: list[str] = []
    requested = 0

    for idisk_id, cell in rows:
        for did in parse_dsld_ids(cell):
            if limit is not None and requested >= limit:
                break
            requested += 1
            try:
                labels.append(client.fetch(did, idisk_id=idisk_id))
            except DSLDNotFound:
                missing.append(did)
        if limit is not None and requested >= limit:
            break

    ids = sorted({str(lb.dsld_id) for lb in labels})
    manifest = {
        "source": "dsld",
        "api": DSLD_API,
        "executed_at": started,
        "requested": requested,
        "fetched": len(labels),
        "missing": missing,
        "limit": limit,
        "dsld_ids": ids,
    }
    return labels, manifest


def save_labels(
    labels: list[Label],
    manifest: dict,
    out_dir: str | Path,
    run_name: str = "dsld",
) -> dict[str, Path]:
    """Write payload (gitignored) and manifest (committable) side by side."""
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    man_dir = out_dir / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{run_name}.jsonl"
    with raw_path.open("w") as fh:
        for lb in labels:
            fh.write(json.dumps(asdict(lb)) + "\n")

    man_path = man_dir / f"{run_name}.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    return {"raw": raw_path, "manifest": man_path}
