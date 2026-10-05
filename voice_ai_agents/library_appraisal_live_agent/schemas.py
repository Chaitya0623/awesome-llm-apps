"""Structured data for the AI Library Appraisal Agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BookFormat = Literal["hardcover", "paperback", "mass_market", "unknown"]
ItemCategory = Literal[
    "furniture", "shelving", "art", "decor", "lighting", "electronics",
    "appliance", "textile", "consumable", "other",
]
EntryStatus = Literal["pending", "pricing", "priced", "failed"]
Routing = Literal["standard_contents", "specialist_review", "needs_more_evidence"]


class Locale(BaseModel):
    city: str = ""
    country: str = ""
    currency: str = "USD"

    @property
    def label(self) -> str:
        return ", ".join(part for part in (self.city, self.country) if part) or "not specified"

    @property
    def is_set(self) -> bool:
        return bool(self.city or self.country)


class SpineReading(BaseModel):
    title: str = Field(description="Title as printed on the spine, corrected for obvious OCR slips.")
    author: str = Field(default="", description="Author surname or full name if visible.")
    publisher: str = Field(default="", description="Publisher or imprint if visible.")
    format: BookFormat = "unknown"
    confidence: float = Field(default=0.5, description="0-1 confidence the title is read correctly.")


class ItemReading(BaseModel):
    category: ItemCategory
    name: str = Field(description="Short generic name, e.g. 'oak bookcase', 'framed oil portrait', 'espresso machine'.")
    description: str = Field(default="", description="Material, brand, style, or signature that drives value.")
    quantity: int = 1
    condition: str = Field(default="good", description="new, good, worn, or damaged")
    size_hint: str = Field(default="", description="Approximate dimensions, e.g. '180x90x30 cm'.")


class RoomView(BaseModel):
    visible: bool = Field(description="True when enough of the room is visible to estimate its dimensions.")
    width_m: float = 0.0
    depth_m: float = 0.0
    ceiling_m: float = 0.0
    shelf_units_visible: int = 0
    shelf_linear_m_visible: float = Field(default=0.0, description="Running length of shelf boards visible, in metres.")
    cues_used: list[str] = Field(default_factory=list, description="Known-size objects used for scale, e.g. 'door 2.0 m'.")


class FrameScan(BaseModel):
    """One camera frame read by the vision model."""

    spines: list[SpineReading] = Field(default_factory=list)
    items: list[ItemReading] = Field(default_factory=list)
    room: RoomView = Field(default_factory=lambda: RoomView(visible=False))


class PriceEstimate(BaseModel):
    low: float
    mid: float
    high: float
    currency: str
    basis: str = ""
    sources: list[str] = Field(default_factory=list)
    collectible: bool = False


class Book(BaseModel):
    id: str
    title: str
    author: str = ""
    publisher: str = ""
    format: BookFormat = "unknown"
    confidence: float = 0.5
    sightings: int = 1
    frame_ids: list[str] = Field(default_factory=list)
    price: PriceEstimate | None = None
    status: EntryStatus = "pending"


class NonBookItem(BaseModel):
    id: str
    category: ItemCategory
    name: str
    description: str = ""
    quantity: int = 1
    condition: str = "good"
    size_hint: str = ""
    frame_ids: list[str] = Field(default_factory=list)
    price: PriceEstimate | None = None
    status: EntryStatus = "pending"


class RoomMeasurement(BaseModel):
    width_m: float = 0.0
    depth_m: float = 0.0
    ceiling_m: float = 0.0
    floor_m2: float = 0.0
    wall_m2: float = 0.0
    total_surface_m2: float = 0.0
    shelf_linear_m: float = 0.0
    shelf_units: int = 0
    range_pct: float = 0.0
    samples: int = 0
    calibrated: bool = False
    calibration_note: str = ""
    cues_used: list[str] = Field(default_factory=list)


class AppraisalRequest(BaseModel):
    """A typed or pasted library description, normalized for the appraisal graph (ADK Web path)."""

    city: str = "not specified"
    country: str = "not specified"
    currency: str = Field(default="", description="ISO 4217 code for the location, e.g. INR, EUR, USD.")
    books: list[SpineReading] = Field(default_factory=list)
    items: list[ItemReading] = Field(default_factory=list)
    room: RoomView = Field(default_factory=lambda: RoomView(visible=False))
    room_dimensions_stated: bool = Field(
        default=False, description="True when the room dimensions came from the claimant, not a guess."
    )


class UnderwritingReview(BaseModel):
    routing: Routing
    routing_reason: str
    specialist_referrals: list[str] = Field(default_factory=list)
    schedule_separately: list[str] = Field(default_factory=list)
    low_confidence_reads: list[str] = Field(default_factory=list)
    unpriced: list[str] = Field(default_factory=list)
    measurement_notes: list[str] = Field(default_factory=list)


class AppraisalPacket(BaseModel):
    title: str
    locale: str
    currency: str
    totals: dict
    routing: Routing
    markdown: str
