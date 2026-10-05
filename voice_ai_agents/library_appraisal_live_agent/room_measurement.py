"""Deterministic room surface-area estimation from per-frame reference-scale readings.

Each camera frame gives a rough RoomView (the vision model sizes the room from
objects of known size). We take medians across frames to damp outliers, then
optionally rescale everything by a calibration factor derived from one real
measurement the claimant confirms ("that bookcase is 2.1 m tall").
"""

from __future__ import annotations

from statistics import median

try:
    from .schemas import RoomMeasurement, RoomView
except ImportError:
    from schemas import RoomMeasurement, RoomView

# Reference sizes the vision prompt leans on for scale.
KNOWN_SIZES_M = {
    "trade paperback spine height": 0.215,
    "mass-market paperback spine height": 0.175,
    "standard hardcover spine height": 0.24,
    "interior door height": 2.0,
    "interior door width": 0.8,
    "bookshelf depth": 0.30,
    "shelf spacing (vertical)": 0.30,
    "light switch height from floor": 1.2,
    "dining/coffee table height": 0.45,
    "standard chair seat height": 0.45,
}

AVG_BOOK_THICKNESS_M = 0.028
SHELF_FILL_RATIO = 0.85
UNCALIBRATED_RANGE = 0.25
CALIBRATED_RANGE = 0.10
CALIBRATION_BOUNDS = (0.5, 2.0)


def known_sizes_prompt() -> str:
    return "\n".join(f"- {k}: {v:.2f} m" for k, v in KNOWN_SIZES_M.items())


def calibration_factor(estimated_m: float, real_m: float) -> float:
    """Ratio to scale linear estimates by; clamped so one bad reading can't wreck the room."""
    if estimated_m <= 0 or real_m <= 0:
        return 1.0
    lo, hi = CALIBRATION_BOUNDS
    return max(lo, min(hi, real_m / estimated_m))


def _spread(values: list[float]) -> float:
    """Relative median absolute deviation: robust to the odd wild frame, like the medians themselves."""
    if len(values) < 2:
        return 0.0
    m = median(values)
    return median(abs(v - m) for v in values) / m if m else 0.0


def estimate_room(
    views: list[RoomView],
    book_count: int = 0,
    factor: float = 1.0,
    calibration_note: str = "",
) -> RoomMeasurement:
    usable = [v for v in views if v.visible]
    widths = [v.width_m for v in usable if v.width_m > 0]
    depths = [v.depth_m for v in usable if v.depth_m > 0]
    ceilings = [v.ceiling_m for v in usable if v.ceiling_m > 0]

    w = median(widths) * factor if widths else 0.0
    d = median(depths) * factor if depths else 0.0
    h = median(ceilings) * factor if ceilings else 0.0

    floor = w * d
    walls = 2 * (w + d) * h
    # Floor + ceiling + walls: the full interior surface an adjuster would quote for refinishing.
    total = 2 * floor + walls

    # Shelf length: the widest single-frame reading vs. what the book count implies.
    frame_shelf = max((v.shelf_linear_m_visible for v in views), default=0.0) * factor
    book_shelf = book_count * AVG_BOOK_THICKNESS_M / SHELF_FILL_RATIO
    shelf_units = max((v.shelf_units_visible for v in views), default=0)

    calibrated = factor != 1.0 or bool(calibration_note)
    base = CALIBRATED_RANGE if calibrated else UNCALIBRATED_RANGE
    spread = max(_spread(widths), _spread(depths), _spread(ceilings))
    range_pct = round(min(0.6, base + spread / 2), 3)

    cues = sorted({c for v in usable for c in v.cues_used})
    return RoomMeasurement(
        width_m=round(w, 2),
        depth_m=round(d, 2),
        ceiling_m=round(h, 2),
        floor_m2=round(floor, 1),
        wall_m2=round(walls, 1),
        total_surface_m2=round(total, 1),
        shelf_linear_m=round(max(frame_shelf, book_shelf), 1),
        shelf_units=shelf_units,
        range_pct=range_pct,
        samples=len(usable),
        calibrated=calibrated,
        calibration_note=calibration_note,
        cues_used=cues[:12],
    )
