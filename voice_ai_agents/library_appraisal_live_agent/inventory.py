"""In-memory library inventory for one appraisal, de-duplicated across camera frames.

A sweep sees the same spine in many consecutive frames, so every reading is
merged into an existing entry when it fuzzy-matches one already on file. The
inventory round-trips through plain dicts so the ADK graph can read it from
session state.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from itertools import count
from typing import Any

try:
    from .room_measurement import estimate_room
    from .schemas import (
        Book, ItemReading, Locale, NonBookItem, PriceEstimate, RoomMeasurement, RoomView, SpineReading,
    )
except ImportError:
    from room_measurement import estimate_room
    from schemas import (
        Book, ItemReading, Locale, NonBookItem, PriceEstimate, RoomMeasurement, RoomView, SpineReading,
    )

BOOK_MATCH = 0.88
ITEM_MATCH = 0.85
MIN_SPINE_CONFIDENCE = 0.35
_ARTICLES = re.compile(r"^(the|a|an)\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    cleaned = " ".join(_NON_ALNUM.sub(" ", text.lower()).split())
    return _ARTICLES.sub("", cleaned)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def _authors_agree(a: str, b: str) -> bool:
    """Lenient: spines often show only a surname ("Rothfuss" vs "Patrick Rothfuss")."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb or na in nb or nb in na:
        return True
    return na.split()[-1] == nb.split()[-1] or _similar(na, nb) >= 0.7


class InventoryStore:
    def __init__(self, locale: Locale | None = None):
        self.locale = locale or Locale()
        self.books: dict[str, Book] = {}
        self.items: dict[str, NonBookItem] = {}
        self.room_views: list[RoomView] = []
        self.calibration_factor = 1.0
        self.calibration_note = ""
        self.notes: list[dict[str, str]] = []  # claimant remarks pinned to evidence frames
        self.frames_scanned = 0
        self.revision = 0
        self._ids = count(1)

    # ------------------------------------------------------------ ingest

    def add_spine(self, reading: SpineReading, frame_id: str) -> Book | None:
        if not reading.title.strip() or reading.confidence < MIN_SPINE_CONFIDENCE:
            return None
        for book in self.books.values():
            if _similar(book.title, reading.title) >= BOOK_MATCH and _authors_agree(book.author, reading.author):
                book.sightings += 1
                if frame_id not in book.frame_ids:
                    book.frame_ids.append(frame_id)
                # Keep the clearest reading of the spine.
                if reading.confidence > book.confidence:
                    book.title, book.confidence = reading.title, reading.confidence
                book.author = book.author or reading.author
                book.publisher = book.publisher or reading.publisher
                if book.format == "unknown":
                    book.format = reading.format
                self.revision += 1
                return book
        book = Book(
            id=f"B{next(self._ids):04d}", title=reading.title.strip(), author=reading.author.strip(),
            publisher=reading.publisher.strip(), format=reading.format, confidence=reading.confidence,
            frame_ids=[frame_id],
        )
        self.books[book.id] = book
        self.revision += 1
        return book

    def add_item(self, reading: ItemReading, frame_id: str) -> NonBookItem:
        for item in self.items.values():
            if item.category == reading.category and _similar(item.name, reading.name) >= ITEM_MATCH:
                # The same object seen again: never sum quantities across frames.
                item.quantity = max(item.quantity, reading.quantity)
                item.description = item.description or reading.description
                item.size_hint = item.size_hint or reading.size_hint
                if frame_id not in item.frame_ids:
                    item.frame_ids.append(frame_id)
                self.revision += 1
                return item
        item = NonBookItem(
            id=f"I{next(self._ids):04d}", category=reading.category, name=reading.name.strip(),
            description=reading.description, quantity=max(1, reading.quantity), condition=reading.condition,
            size_hint=reading.size_hint, frame_ids=[frame_id],
        )
        self.items[item.id] = item
        self.revision += 1
        return item

    def add_room_view(self, view: RoomView) -> None:
        self.room_views.append(view)
        self.revision += 1

    def set_locale(self, locale: Locale) -> bool:
        """Returns True when the pricing location changed, which resets every valuation."""
        changed = (locale.city, locale.country, locale.currency) != (
            self.locale.city, self.locale.country, self.locale.currency,
        )
        self.locale = locale
        if changed:
            for entry in self.entries():
                entry.price, entry.status = None, "pending"
            self.revision += 1
        return changed

    def set_calibration(self, factor: float, note: str) -> None:
        self.calibration_factor = factor
        self.calibration_note = note
        self.revision += 1

    # ------------------------------------------------------------ pricing

    def entries(self) -> list[Book | NonBookItem]:
        return [*self.books.values(), *self.items.values()]

    def pending_count(self) -> int:
        return sum(entry.status in {"pending", "pricing"} for entry in self.entries())

    def mark_pricing(self, ids: set[str]) -> None:
        for entry in self.entries():
            if entry.id in ids and entry.status == "pending":
                entry.status = "pricing"

    def apply_valuations(self, valuations: dict[str, Any]) -> None:
        """Merge ApplyValuations output from the ADK graph back into the live inventory."""
        currency = valuations.get("currency")
        if currency and currency != self.locale.currency:
            return  # The location changed while this batch was being priced.
        priced = {**valuations.get("books", {}), **valuations.get("items", {})}
        failed = set(valuations.get("failed", []))
        for entry in self.entries():
            if entry.id in priced:
                entry.price, entry.status = PriceEstimate.model_validate(priced[entry.id]), "priced"
            elif entry.id in failed:
                entry.status = "failed"
            elif entry.status == "pricing":
                entry.status = "pending"
        self.revision += 1

    def retry_failed(self) -> int:
        failed = [entry for entry in self.entries() if entry.status == "failed"]
        for entry in failed:
            entry.status = "pending"
        self.revision += bool(failed)
        return len(failed)

    # ------------------------------------------------------------ views

    def room(self) -> RoomMeasurement:
        return estimate_room(self.room_views, len(self.books), self.calibration_factor, self.calibration_note)

    def to_state(self) -> dict[str, Any]:
        return {
            "locale": self.locale.model_dump(),
            "books": [book.model_dump() for book in self.books.values()],
            "items": [item.model_dump() for item in self.items.values()],
            "room_views": [view.model_dump() for view in self.room_views],
            "calibration": {"factor": self.calibration_factor, "note": self.calibration_note},
            "notes": list(self.notes),
            "frames_scanned": self.frames_scanned,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "InventoryStore":
        store = cls(Locale.model_validate(state.get("locale") or {}))
        for raw in state.get("books", []):
            book = Book.model_validate(raw)
            store.books[book.id] = book
        for raw in state.get("items", []):
            item = NonBookItem.model_validate(raw)
            store.items[item.id] = item
        store.room_views = [RoomView.model_validate(v) for v in state.get("room_views", [])]
        calibration = state.get("calibration") or {}
        store.calibration_factor = float(calibration.get("factor", 1.0))
        store.calibration_note = str(calibration.get("note", ""))
        store.notes = list(state.get("notes", []))
        store.frames_scanned = int(state.get("frames_scanned", 0))
        used = [int(entry_id[1:]) for entry_id in [*store.books, *store.items] if entry_id[1:].isdigit()]
        store._ids = count(max(used, default=0) + 1)
        return store
