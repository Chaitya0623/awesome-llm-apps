"""Deterministic valuation, measurement, underwriting, and packet builders."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from typing import Any

try:
    from .inventory import InventoryStore
    from .schemas import (
        AppraisalPacket, AppraisalRequest, Locale, PriceEstimate, RoomView, UnderwritingReview,
    )
except ImportError:
    from inventory import InventoryStore
    from schemas import (
        AppraisalPacket, AppraisalRequest, Locale, PriceEstimate, RoomView, UnderwritingReview,
    )

MAX_BOOKS_PER_RUN = 20
MAX_ITEMS_PER_RUN = 10
LOW_CONFIDENCE_READ = 0.6
SCHEDULE_SHARE = 0.10  # one line worth >=10% of the contents total gets listed for a scheduled rider
MIN_ROOM_SAMPLES = 3
UNPRICED_SHARE_LIMIT = 0.25
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_NOT_SPECIFIED = {"", "not specified", "unknown"}


# ------------------------------------------------------------------ intake

def inventory_from_request(request: dict[str, Any]) -> dict[str, Any]:
    """Seed an inventory from a typed library description (the ADK Web path)."""
    req = AppraisalRequest.model_validate(request)
    city = "" if req.city.lower() in _NOT_SPECIFIED else req.city
    country = "" if req.country.lower() in _NOT_SPECIFIED else req.country
    currency = req.currency.upper() if re.fullmatch(r"[A-Za-z]{3}", req.currency or "") else "USD"
    store = InventoryStore(Locale(city=city, country=country, currency=currency))
    for spine in req.books:
        store.add_spine(spine, "typed")
    for item in req.items:
        store.add_item(item, "typed")
    if req.room.visible or req.room.width_m:
        store.add_room_view(RoomView.model_validate({**req.room.model_dump(), "visible": True}))
    if req.room_dimensions_stated:
        store.set_calibration(1.0, "room dimensions stated by claimant")
    return store.to_state()


def select_pricing_queue(inventory: dict[str, Any]) -> dict[str, Any]:
    """Next batch of entries to value. Empty until a pricing location is known."""
    locale = Locale.model_validate(inventory.get("locale") or {})
    if not locale.is_set:
        return {"locale": locale.model_dump(), "books": [], "items": []}
    waiting = lambda e: e.get("status") in {"pending", "pricing"}
    books = [
        {k: b.get(k, "") for k in ("id", "title", "author", "publisher", "format")}
        for b in inventory.get("books", []) if waiting(b)
    ][:MAX_BOOKS_PER_RUN]
    items = [
        {k: i.get(k, "") for k in ("id", "name", "category", "description", "size_hint", "condition")}
        for i in inventory.get("items", []) if waiting(i)
    ][:MAX_ITEMS_PER_RUN]
    return {"locale": locale.model_dump(), "books": books, "items": items}


# ------------------------------------------------------------------ valuation

def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Pull the first JSON array out of a model reply that may include fences or chatter."""
    text = str(text or "")
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [row for row in data if isinstance(row, dict)]


def to_estimate(row: dict[str, Any], currency: str, fallback_sources: list[str]) -> PriceEstimate | None:
    try:
        low, mid, high = (float(row[key]) for key in ("low", "mid", "high"))
    except (KeyError, TypeError, ValueError):
        return None
    if mid <= 0:
        return None
    low, mid, high = sorted((max(0.0, low), mid, high))
    sources = [s for s in row.get("sources") or [] if isinstance(s, str) and s.startswith("http")]
    return PriceEstimate(
        low=low, mid=mid, high=high, currency=currency,
        basis=str(row.get("basis", ""))[:200],
        sources=(sources or fallback_sources)[:3],
        collectible=bool(row.get("collectible", False)),
    )


def apply_valuations(
    queue: dict[str, Any],
    book_quotes: Any,
    item_quotes: Any,
    book_sources: list[str] | None = None,
    item_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Turn the pricing agents' replies into estimates keyed by entry id; anything unquoted fails."""
    currency = Locale.model_validate(queue.get("locale") or {}).currency
    result: dict[str, Any] = {"currency": currency, "books": {}, "items": {}, "failed": []}
    for kind, quotes, sources in (("books", book_quotes, book_sources), ("items", item_quotes, item_sources)):
        rows = extract_json_array(quotes) if isinstance(quotes, str) else list(quotes or [])
        by_id = {str(row.get("id")): row for row in rows}
        for position, entry in enumerate(queue.get(kind, [])):
            row = by_id.get(entry["id"]) or (rows[position] if position < len(rows) and "id" not in rows[position] else None)
            estimate = to_estimate(row, currency, sources or []) if row else None
            if estimate:
                result[kind][entry["id"]] = estimate.model_dump()
            else:
                result["failed"].append(entry["id"])
    return result


def merge_valuations(inventory: dict[str, Any], valuations: dict[str, Any]) -> dict[str, Any]:
    store = InventoryStore.from_state(inventory)
    store.apply_valuations(valuations)
    return store.to_state()


def inventory_totals(inventory: dict[str, Any]) -> dict[str, Any]:
    def accumulate(entries, use_quantity: bool) -> dict[str, Any]:
        low = mid = high = 0.0
        priced = 0
        for entry in entries:
            price = entry.get("price")
            if not price:
                continue
            qty = entry.get("quantity", 1) if use_quantity else 1
            low += price["low"] * qty
            mid += price["mid"] * qty
            high += price["high"] * qty
            priced += 1
        return {"count": len(entries), "priced": priced, "low": round(low, 2), "mid": round(mid, 2), "high": round(high, 2)}

    books = accumulate(inventory.get("books", []), use_quantity=False)
    items = accumulate(inventory.get("items", []), use_quantity=True)
    return {
        "currency": (inventory.get("locale") or {}).get("currency", "USD"),
        "books": books,
        "items": items,
        "grand": {key: round(books[key] + items[key], 2) for key in ("low", "mid", "high")},
    }


def measure_room(inventory: dict[str, Any]) -> dict[str, Any]:
    return InventoryStore.from_state(inventory).room().model_dump()


# ------------------------------------------------------------------ underwriting

def _line_value(entry: dict[str, Any]) -> float:
    price = entry.get("price")
    return price["mid"] * entry.get("quantity", 1) if price else 0.0


def underwriting_review(inventory: dict[str, Any], room: dict[str, Any]) -> dict[str, Any]:
    books, items = inventory.get("books", []), inventory.get("items", [])
    total = inventory_totals(inventory)["grand"]["mid"]

    specialist = [f"{b['title']} ({b.get('author') or 'unknown author'}): possible collectible edition"
                  for b in books if (b.get("price") or {}).get("collectible")]
    specialist += [f"{i['name']}: original art needs an independent appraisal"
                   for i in items if i.get("category") == "art" and "print" not in (i.get("description") or "").lower()
                   and i.get("price")]
    schedule = [
        f"{e.get('title') or e.get('name')}: {_line_value(e):,.0f} {(e.get('price') or {}).get('currency', '')}"
        for e in [*books, *items]
        if total > 0 and _line_value(e) >= SCHEDULE_SHARE * total and len(books) + len(items) > 1
    ]
    low_reads = [f"{b['title']} (read confidence {b.get('confidence', 0):.0%})"
                 for b in books if b.get("confidence", 1) < LOW_CONFIDENCE_READ]
    unpriced = [e.get("title") or e.get("name") for e in [*books, *items] if e.get("status") != "priced"]

    notes = []
    if room.get("samples", 0) == 0:
        notes.append("No wide view of the room yet; surface area cannot be estimated.")
    elif room.get("samples", 0) < MIN_ROOM_SAMPLES:
        notes.append(f"Room sized from only {room['samples']} view(s); a slower wide pass will tighten the estimate.")
    if room.get("samples", 0) and not room.get("calibrated"):
        notes.append("Measurements are uncalibrated; confirm one real dimension (e.g. a door or bookcase height).")

    entries = len(books) + len(items)
    if specialist:
        routing, reason = "specialist_review", "Possible collectibles or original art need a specialist appraiser."
    elif entries == 0:
        routing, reason = "needs_more_evidence", "No books or contents have been captured yet."
    elif len(unpriced) > UNPRICED_SHARE_LIMIT * entries:
        routing, reason = "needs_more_evidence", "Too many entries are still unpriced for a reliable total."
    elif room.get("samples", 0) == 0:
        routing, reason = "needs_more_evidence", "The room has not been measured."
    else:
        routing, reason = "standard_contents", "Inventory is priced and measured; standard contents review."

    return UnderwritingReview(
        routing=routing, routing_reason=reason, specialist_referrals=specialist,
        schedule_separately=schedule, low_confidence_reads=low_reads, unpriced=unpriced,
        measurement_notes=notes,
    ).model_dump()


# ------------------------------------------------------------------ packet

def build_appraisal_packet(inventory: dict[str, Any], room: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    locale = Locale.model_validate(inventory.get("locale") or {})
    totals = inventory_totals(inventory)
    cur = totals["currency"]
    money = lambda value: f"{value:,.0f} {cur}"
    pct = round(room.get("range_pct", 0) * 100)
    t_books, t_items, grand = totals["books"], totals["items"], totals["grand"]

    lines = [
        "# Library Contents Appraisal",
        "",
        f"- Generated: {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}",
        f"- Pricing location: {locale.label} ({cur})",
        f"- Frames analysed: {inventory.get('frames_scanned', 0)}",
        f"- Routing: **{review['routing'].replace('_', ' ')}**: {review['routing_reason']}",
        "",
        "## Valuation summary",
        "",
        "| | Count | Priced | Low | Mid | High |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Books | {t_books['count']} | {t_books['priced']} | {money(t_books['low'])} | {money(t_books['mid'])} | {money(t_books['high'])} |",
        f"| Non-book items | {t_items['count']} | {t_items['priced']} | {money(t_items['low'])} | {money(t_items['mid'])} | {money(t_items['high'])} |",
        f"| **Total** | | | **{money(grand['low'])}** | **{money(grand['mid'])}** | **{money(grand['high'])}** |",
        "",
        "## Room measurements",
        "",
        f"- Dimensions: {room.get('width_m', 0)} m x {room.get('depth_m', 0)} m, ceiling {room.get('ceiling_m', 0)} m",
        f"- Floor area: {room.get('floor_m2', 0)} m² (±{pct}%)",
        f"- Wall area (gross, openings not deducted): {room.get('wall_m2', 0)} m² (±{pct}%)",
        f"- Total interior surface (floor + ceiling + walls): {room.get('total_surface_m2', 0)} m²",
        f"- Shelving: {room.get('shelf_linear_m', 0)} linear m across ~{room.get('shelf_units', 0)} units",
        f"- Calibrated: {'yes, ' + room['calibration_note'] if room.get('calibrated') else 'no (reference-object scale only)'}",
        f"- Scale cues: {', '.join(room.get('cues_used') or []) or 'none recorded'}",
        "",
    ]
    sections = [
        ("Refer to specialist appraiser", review["specialist_referrals"]),
        ("Consider scheduling separately", review["schedule_separately"]),
        ("Low-confidence spine reads", review["low_confidence_reads"]),
        ("Not yet priced", review["unpriced"]),
        ("Measurement notes", review["measurement_notes"]),
        ("Claimant notes", [f"{n['note']} (evidence/{n['frame']}.jpg)" for n in inventory.get("notes", [])]),
    ]
    for heading, entries in sections:
        if entries:
            lines += [f"## {heading}", "", *[f"- {entry}" for entry in entries], ""]
    lines += [
        "## Method and limitations",
        "",
        "- Titles were read from spines on camera. Unreadable spines are not listed, so the book count is a lower bound.",
        "- Prices are AI-assisted replacement-cost estimates grounded in web search for the location above;",
        "  they are not a certified appraisal. Sources are listed per line in books.csv and items.csv.",
        "- Room dimensions are estimated from objects of known size in the video and are approximate.",
        "- This packet does not confirm coverage. Coverage is decided by the insurer.",
    ]
    return AppraisalPacket(
        title="Library Contents Appraisal", locale=locale.label, currency=cur, totals=totals,
        routing=review["routing"], markdown="\n".join(lines),
    ).model_dump()


def _csv(header: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _price_cols(entry: dict[str, Any], qty: int = 1) -> list[Any]:
    price = entry.get("price")
    if not price:
        return ["", "", "", "", "", ""]
    return [price["low"], price["mid"], price["high"], price["currency"], round(price["mid"] * qty, 2), price.get("basis", "")]


def books_csv(inventory: dict[str, Any]) -> str:
    return _csv(
        ["id", "title", "author", "publisher", "format", "read_confidence", "low", "mid", "high", "currency",
         "line_mid", "basis", "collectible", "sources", "evidence_frames"],
        [[b["id"], b["title"], b.get("author", ""), b.get("publisher", ""), b.get("format", ""), b.get("confidence", ""),
          *_price_cols(b), (b.get("price") or {}).get("collectible", ""),
          " ".join((b.get("price") or {}).get("sources", [])), " ".join(b.get("frame_ids", []))]
         for b in inventory.get("books", [])],
    )


def items_csv(inventory: dict[str, Any]) -> str:
    return _csv(
        ["id", "category", "name", "description", "quantity", "condition", "size", "unit_low", "unit_mid",
         "unit_high", "currency", "line_mid", "basis", "sources", "evidence_frames"],
        [[i["id"], i["category"], i["name"], i.get("description", ""), i.get("quantity", 1), i.get("condition", ""),
          i.get("size_hint", ""), *_price_cols(i, i.get("quantity", 1)),
          " ".join((i.get("price") or {}).get("sources", [])), " ".join(i.get("frame_ids", []))]
         for i in inventory.get("items", [])],
    )
