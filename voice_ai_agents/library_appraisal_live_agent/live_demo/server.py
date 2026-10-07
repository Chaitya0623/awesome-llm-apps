"""FastAPI backend for the Library Appraisal Live Agent UI.

The browser transport, camera sweep, and frame scanning live here. Valuation,
measurement, underwriting review, and packet generation live in agent.py,
which defines and runs the ADK graph.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import secrets
import sys
import time
import uuid
import zipfile
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _load_dotenv() -> None:
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _cors_origins() -> list[str]:
    raw = os.getenv("APPRAISAL_CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return ["http://127.0.0.1:4178", "http://localhost:4178"]


_load_dotenv()

from agent import MODEL, OPENAI_PRICING_MODEL, PRICING_PROVIDER, run_appraisal_workflow  # noqa: E402
from appraisal_rules import books_csv, inventory_totals, items_csv, underwriting_review  # noqa: E402
from inventory import InventoryStore, clean_item_readings  # noqa: E402
from room_measurement import calibration_factor, known_sizes_prompt  # noqa: E402
from schemas import FrameScan, Locale  # noqa: E402

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from live_tools import (  # noqa: E402
    LIVE_MODEL_ID,
    OPENAI_SKETCH_MODEL,
    OPENAI_VISION_MODEL,
    SKETCH_MODEL_ID,
    SKETCH_PROVIDER,
    TOOL_NAMES,
    VISION_MODEL_ID,
    VISION_PROVIDER,
    build_live_config,
    camera_mode_instruction,
    floor_plan_prompt,
    frame_scan_prompt,
    object_size_prompt,
    scheduling_for,
    summarize_workflow_for_voice,
    tool_headline,
)

GENAI_CLIENT = None
AVATAR_CLIENT = None
OPENAI_CLIENT = None
logger = logging.getLogger(__name__)
FRAME_MAX_AGE_SECONDS = 12.0
SWEEP_INTERVAL_SECONDS = 1.5
SIMILAR_FRAME_THRESHOLD = 6.0  # mean abs pixel difference (0-255) on a 32x24 thumbnail
MAX_EVIDENCE_FRAMES = 200
MAX_VALUATION_RUNS = 25
SYNC_TIMEOUT_SECONDS = 90
SKETCH_TIMEOUT_SECONDS = 120  # image models take 30-60 s for a labelled plan


def avatar_settings() -> dict[str, str]:
    """Keep the optional Cloud avatar transport separate from appraisal model auth."""
    return {
        "name": os.getenv("APPRAISAL_AVATAR_NAME", "").strip(),
        "project": os.getenv("APPRAISAL_AVATAR_PROJECT", "").strip(),
        "location": os.getenv("APPRAISAL_AVATAR_LOCATION", "us-central1").strip(),
        "image": os.getenv("APPRAISAL_AVATAR_IMAGE", "").strip(),
        "voice": os.getenv("APPRAISAL_AVATAR_VOICE", "Kore").strip(),
    }


def avatar_description(enabled: bool | None = None) -> dict[str, Any]:
    settings = avatar_settings()
    configured = bool(settings["project"] and (settings["name"] or settings["image"]))
    return {
        "enabled": configured if enabled is None else enabled,
        "name": "Appraiser" if settings["image"] else settings["name"],
    }


def avatar_reference() -> bytes:
    path = (APP_DIR / avatar_settings()["image"]).resolve()
    data = path.read_bytes()
    if len(data) >= 5 * 1024 * 1024 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Custom avatar must be a PNG under 5 MB")
    width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if width < 704 or height < 1280:
        raise ValueError("Custom avatar must be at least 704 x 1280")
    return data


def live_media_message(blob):
    """Never interpret the avatar's muxed video/voice bytes as raw PCM."""
    mime = blob.mime_type or ""
    if not isinstance(blob.data, bytes) or not mime.startswith(("video/mp4", "audio/pcm")):
        return None
    return {
        "type": "avatar_video" if mime.startswith("video/") else "audio",
        "data": base64.b64encode(blob.data).decode("ascii"),
        "mime_type": mime,
    }


class SessionResponse(BaseModel):
    session_id: str
    model: str
    has_api_key: bool
    state: dict[str, Any]


@dataclass
class AppraisalSession:
    session_id: str
    store: InventoryStore = field(default_factory=InventoryStore)
    transcript: list[dict[str, str]] = field(default_factory=list)
    tool_activity: list[dict[str, Any]] = field(default_factory=list)
    evidence_frames: OrderedDict = field(default_factory=OrderedDict)
    last_workflow: dict[str, Any] | None = None
    last_workflow_key: int | None = None
    floor_plan: dict[str, Any] | None = None  # metadata only; the PNG is served over HTTP
    floor_plan_image: bytes = b""
    floor_plan_revision: int = 0
    last_frame: bytes | None = None
    last_frame_at: float = 0.0
    last_frame_id: str = ""
    last_scanned_id: str = ""
    last_scanned_thumb: bytes | None = None
    sweeping: bool = False
    camera_enabled: bool = False
    owner: str = ""
    updated_at: float = field(default_factory=time.monotonic)
    created_at: float = field(default_factory=time.monotonic)
    deleted: bool = False
    workflow_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    scan_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sweep_task: asyncio.Task | None = None
    valuation_task: asyncio.Task | None = None
    notify: Callable[[], Awaitable[None]] | None = None
    live_socket: Any = None
    tasks: set = field(default_factory=set)


sessions: dict[str, AppraisalSession] = {}

app = FastAPI(title="Library Appraisal Live Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _has_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _client():
    global GENAI_CLIENT
    if not _has_api_key():
        raise HTTPException(
            status_code=503,
            detail=(
                "Missing GOOGLE_API_KEY. Add it to "
                f"{APP_DIR / '.env'} and restart the appraisal backend."
            ),
        )
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    try:
        from google import genai
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Missing google-genai package. Run pip install -r requirements.txt.",
        ) from exc
    if GENAI_CLIENT is None:
        GENAI_CLIENT = genai.Client()
    return GENAI_CLIENT


def _live_client(avatar_enabled: bool = False):
    if not avatar_enabled:
        return _client()
    global AVATAR_CLIENT
    if AVATAR_CLIENT is None:
        from google import genai
        settings = avatar_settings()
        AVATAR_CLIENT = genai.Client(
            vertexai=True, project=settings["project"], location=settings["location"],
        )
    return AVATAR_CLIENT


def _track(session: AppraisalSession, task: asyncio.Task) -> asyncio.Task:
    session.tasks.add(task)

    def finished(done: asyncio.Task) -> None:
        session.tasks.discard(done)
        if not done.cancelled() and done.exception():
            logger.error("Background appraisal task failed: %s", type(done.exception()).__name__)

    task.add_done_callback(finished)
    return task


async def _notify(session: AppraisalSession) -> None:
    if session.notify and not session.deleted:
        with contextlib.suppress(Exception):
            await session.notify()


def append_turn(session: AppraisalSession, speaker: str, text: str, turn_id: str | None = None) -> None:
    session.transcript.append({"id": turn_id or uuid.uuid4().hex, "speaker": speaker, "text": text.strip()})
    session.transcript = session.transcript[-200:]


def _ui_state(session: AppraisalSession) -> dict[str, Any]:
    """Everything the notebook UI renders, computed from the live inventory."""
    store = session.store
    inventory = store.to_state()
    room = store.room().model_dump()
    return {
        "locale": {**store.locale.model_dump(), "label": store.locale.label},
        "books": inventory["books"],
        "items": inventory["items"],
        "notes": inventory["notes"],
        "totals": inventory_totals(inventory),
        "room": room,
        "review": underwriting_review(inventory, room),
        "frames_scanned": store.frames_scanned,
        "pending": store.pending_count(),
        "sweeping": session.sweeping,
        "camera_enabled": session.camera_enabled,
        "packet_ready": session.last_workflow is not None,
        "packet_markdown": (session.last_workflow or {}).get("final_markdown", ""),
        "floor_plan": session.floor_plan,
        "tool_activity": session.tool_activity[-12:],
    }


# ------------------------------------------------------------------ vision

def _thumbnail(jpeg: bytes) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as image:
        return image.convert("L").resize((32, 24)).tobytes()


def _frame_distance(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))


# ------------------------------------------------------------------ model providers

def _openai():
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        from openai import AsyncOpenAI

        OPENAI_CLIENT = AsyncOpenAI()
    return OPENAI_CLIENT


def _openai_content(prompt: str, jpeg: bytes | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if jpeg:
        # Spine text is small; high detail keeps it legible.
        content.append({"type": "input_image", "detail": "high",
                        "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")})
    return [{"role": "user", "content": [*content, {"type": "input_text", "text": prompt}]}]


def vision_model_label() -> str:
    return f"openai:{OPENAI_VISION_MODEL}" if VISION_PROVIDER == "openai" else VISION_MODEL_ID


def sketch_model_label() -> str:
    return f"openai:{OPENAI_SKETCH_MODEL}" if SKETCH_PROVIDER == "openai" else SKETCH_MODEL_ID


async def _read_frame(jpeg: bytes) -> FrameScan:
    """One structured reading of spines, items, and room dimensions from a camera frame."""
    prompt = frame_scan_prompt(known_sizes_prompt())
    if VISION_PROVIDER == "openai":
        response = await _openai().responses.parse(
            model=OPENAI_VISION_MODEL, input=_openai_content(prompt, jpeg), text_format=FrameScan,
        )
        return response.output_parsed or FrameScan()
    from google.genai import types

    response = await _client().aio.models.generate_content(
        model=VISION_MODEL_ID,
        contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FrameScan,
            temperature=0.2,
        ),
    )
    return response.parsed if isinstance(response.parsed, FrameScan) else FrameScan.model_validate_json(response.text or "{}")


async def _ask(prompt: str, jpeg: bytes | None = None) -> str:
    """A short plain-text answer, optionally about an image (currency codes, object sizes)."""
    if VISION_PROVIDER == "openai":
        response = await _openai().responses.create(model=OPENAI_VISION_MODEL, input=_openai_content(prompt, jpeg))
        return response.output_text or ""
    from google.genai import types

    contents = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt] if jpeg else prompt
    response = await _client().aio.models.generate_content(model=VISION_MODEL_ID, contents=contents)
    return response.text or ""


async def _draw(prompt: str) -> tuple[bytes, str] | None:
    """Generate one image; returns (bytes, mime type), or None when the model returned no image."""
    if SKETCH_PROVIDER == "openai":
        response = await _openai().images.generate(
            model=OPENAI_SKETCH_MODEL, prompt=prompt, size="1024x1024", quality="medium",
        )
        data = response.data[0].b64_json if response.data else None
        return (base64.b64decode(data), "image/png") if data else None
    from google.genai import types

    response = await _client().aio.models.generate_content(
        model=SKETCH_MODEL_ID, contents=prompt, config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    part = next(
        (
            part
            for candidate in response.candidates or []
            for part in (candidate.content.parts if candidate.content else [])
            if part.inline_data and part.inline_data.data
        ),
        None,
    )
    return (part.inline_data.data, part.inline_data.mime_type or "image/png") if part else None


async def _scan_frame(session: AppraisalSession, frame_id: str, jpeg: bytes, *, force: bool = False) -> dict[str, Any]:
    """Read one frame with the vision model and merge spines, items, and room view into the inventory."""
    async with session.scan_lock:
        thumb = _thumbnail(jpeg)
        if not force and session.last_scanned_thumb is not None and (
            _frame_distance(thumb, session.last_scanned_thumb) < SIMILAR_FRAME_THRESHOLD
        ):
            session.last_scanned_id = frame_id
            return {"skipped": True}
        session.last_scanned_id, session.last_scanned_thumb = frame_id, thumb
        scan = await _read_frame(jpeg)

    store = session.store
    books_before, items_before = len(store.books), len(store.items)
    for spine in scan.spines:
        store.add_spine(spine, frame_id)
    for item in clean_item_readings(scan.items):
        store.add_item(item, frame_id)
    store.add_room_view(scan.room)
    store.frames_scanned += 1
    if scan.spines or scan.items:
        session.evidence_frames[frame_id] = jpeg
        while len(session.evidence_frames) > MAX_EVIDENCE_FRAMES:
            session.evidence_frames.popitem(last=False)
    request_valuation(session)
    await _notify(session)
    return {
        "frame": frame_id,
        "spines_read": len(scan.spines),
        "new_books": len(store.books) - books_before,
        "new_items": len(store.items) - items_before,
        "room_visible": scan.room.visible,
    }


async def _sweep_loop(session: AppraisalSession) -> None:
    while session.sweeping and not session.deleted:
        fresh = session.last_frame and time.monotonic() - session.last_frame_at < FRAME_MAX_AGE_SECONDS
        if fresh and session.last_frame_id != session.last_scanned_id:
            try:
                await _scan_frame(session, session.last_frame_id, session.last_frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep sweeping past a single unreadable frame or API hiccup.
                logger.exception("Frame scan failed")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)


def set_sweeping(session: AppraisalSession, enabled: bool) -> None:
    session.sweeping = enabled
    if enabled and (session.sweep_task is None or session.sweep_task.done()):
        session.sweep_task = _track(session, asyncio.create_task(_sweep_loop(session)))


async def _resolve_currency(city: str, country: str) -> str:
    answer = await _ask(f"ISO 4217 currency code used in {city}, {country}? Reply with only the 3-letter code.")
    code = answer.strip().upper()[:3]
    return code if len(code) == 3 and code.isalpha() else "USD"


async def set_pricing_location(session: AppraisalSession, city: str, country: str, currency: str = "") -> Locale:
    city, country = city.strip(), country.strip()
    currency = currency.strip().upper() or await _resolve_currency(city, country)
    session.store.set_locale(Locale(city=city, country=country, currency=currency))
    request_valuation(session)
    await _notify(session)
    return session.store.locale


async def calibrate_room_scale(session: AppraisalSession, object_name: str, dimension: str, real_size_m: float) -> dict[str, Any]:
    if not session.last_frame or time.monotonic() - session.last_frame_at > FRAME_MAX_AGE_SECONDS:
        return {"calibrated": False, "message": "No recent camera frame. Ask the claimant to show the object."}
    answer = await _ask(object_size_prompt(object_name, dimension, known_sizes_prompt()), session.last_frame)
    try:
        estimated = float(answer.strip().split()[0].rstrip("m"))
    except (ValueError, IndexError):
        estimated = 0.0
    if estimated <= 0 or real_size_m <= 0:
        return {"calibrated": False, "message": f"Could not size the {object_name}. Ask for a wider, steadier view."}
    # The vision estimate is on the uncalibrated scale, so a new calibration replaces the old one.
    note = f"{object_name} {dimension} confirmed {real_size_m:.2f} m (camera estimate {estimated:.2f} m)"
    session.store.set_calibration(calibration_factor(estimated, real_size_m), note)
    await _notify(session)
    return {"calibrated": True, "note": note, "room": session.store.room().model_dump()}


async def set_item_count(session: AppraisalSession, item_name: str, quantity: int) -> dict[str, Any]:
    item = session.store.set_item_count(item_name, quantity)
    if item is None:
        return {
            "updated": False,
            "message": f"No item like '{item_name}' on the ledger yet. Show it on camera, or use one of these names.",
            "known_items": [i.name for i in session.store.items.values()][:20],
        }
    # Prices are per unit and totals multiply by quantity, so nothing needs re-pricing.
    await _notify(session)
    return {"updated": True, "item": item.name, "quantity": item.quantity}


def _room_labels(room: dict[str, Any]) -> dict[str, Any]:
    return {key: room[key] for key in ("width_m", "depth_m", "ceiling_m", "floor_m2", "wall_m2", "calibrated")}


async def draw_floor_plan(session: AppraisalSession, args: dict[str, Any]) -> dict[str, Any]:
    """Sketch a top-down plan with the image model, labelled with the deterministic measurements."""

    layout = str(args.get("layout_description", "")).strip()[:2000]
    trigger = args.get("trigger", "automatic")
    if not layout:
        return {"sketched": False, "message": "Describe where the bookcases and other contents stand."}
    if trigger not in {"automatic", "explicit_request", "correction"}:
        return {"sketched": False, "message": "Unknown floor plan trigger."}
    if trigger == "correction" and not session.floor_plan:
        return {"sketched": False, "message": "There is no floor plan to correct yet."}
    room = session.store.room().model_dump()
    if not room["floor_m2"]:
        return {"sketched": False, "message": "The room is not measured yet. Ask for a slow wide pass of every wall first."}
    labels = _room_labels(room)
    current = session.floor_plan
    if current and current["layout"].casefold() == layout.casefold() and current["room"] == labels:
        return {"sketched": True, "reused": True, "version": current["version"], "message": "The current plan already shows this layout."}
    session.floor_plan_revision += 1
    request_revision = session.floor_plan_revision
    image = await asyncio.wait_for(
        _draw(floor_plan_prompt(layout, room, session.store.to_state()["items"])), SKETCH_TIMEOUT_SECONDS,
    )
    if image is None:
        return {"sketched": False, "message": "The sketch model returned no image. Continue without it."}
    if session.deleted or request_revision != session.floor_plan_revision:
        return {"sketched": False, "message": "Superseded by a newer floor plan request."}
    session.floor_plan_image, mime_type = image
    session.floor_plan = {
        "version": request_revision,
        "mime_type": mime_type,
        "layout": layout,
        "trigger": trigger,
        "room": labels,
    }
    await _notify(session)
    return {
        "sketched": True,
        "version": request_revision,
        "next_step": "Say the plan is an illustration with the measured sizes and ask if the layout looks right.",
    }


# ------------------------------------------------------------------ ADK appraisal graph

async def _run_workflow_cached(session: AppraisalSession) -> dict[str, Any]:
    async with session.workflow_lock:
        key = session.store.revision
        if session.last_workflow is not None and session.last_workflow_key == key:
            return session.last_workflow
        snapshot = session.store.to_state()
        queued = {e["id"] for e in [*snapshot["books"], *snapshot["items"]] if e["status"] == "pending"}
        session.store.mark_pricing(queued)
        await _notify(session)
        try:
            workflow = await run_appraisal_workflow(snapshot, session_id=session.session_id)
        except Exception:
            session.store.apply_valuations({"currency": session.store.locale.currency, "failed": sorted(queued)})
            raise
        session.store.apply_valuations(workflow["valuations"])
        session.last_workflow = workflow
        # Entries scanned while this run was pricing are still pending: leave the cache stale so the
        # valuation loop runs again for them instead of returning this result forever.
        session.last_workflow_key = None if session.store.pending_count() else session.store.revision
        return workflow


async def _valuation_loop(session: AppraisalSession) -> dict[str, Any] | None:
    """Keep running the graph while the inventory has entries waiting for a local price."""
    workflow = None
    for _ in range(MAX_VALUATION_RUNS):
        if session.deleted:
            break
        workflow = await _run_workflow_cached(session)
        await _notify(session)
        if not session.store.locale.is_set or not session.store.pending_count():
            break
    return workflow


def request_valuation(session: AppraisalSession) -> asyncio.Task | None:
    if not session.store.locale.is_set or not session.store.pending_count() or not _has_api_key():
        return session.valuation_task
    if session.valuation_task is None or session.valuation_task.done():
        session.valuation_task = _track(session, asyncio.create_task(_valuation_loop(session)))
    return session.valuation_task


async def sync_appraisal_packet(session: AppraisalSession) -> dict[str, Any]:
    session.store.retry_failed()
    task = request_valuation(session)
    if task is not None and not task.done():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), SYNC_TIMEOUT_SECONDS)
    workflow = await _run_workflow_cached(session)
    return summarize_workflow_for_voice(workflow)


# ------------------------------------------------------------------ HTTP

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL,
        "has_api_key": _has_api_key(),
        "live_model": LIVE_MODEL_ID,
        "vision_model": vision_model_label(),
        "sketch_model": sketch_model_label(),
        "pricing": f"openai:{OPENAI_PRICING_MODEL}" if PRICING_PROVIDER == "openai" else f"gemini:{MODEL}",
        "tools": TOOL_NAMES,
        "avatar": avatar_description(),
    }


SESSION_TTL = 60 * 60
MAX_SESSIONS = 32
MAX_MESSAGE_BYTES = 800000


def allowed_origin(origin: str | None, host: str, scheme="http") -> bool:
    if not origin:
        return False
    return origin in set(_cors_origins()) | {f"{scheme}://{host}"}


def local_host(host: str) -> bool:
    return host.split(":")[0] in {"localhost", "127.0.0.1"}


@app.middleware("http")
async def local_access(request: Request, call_next):
    # Local-only demo: public serving requires a separate authenticated deployment design.
    if not local_host(request.headers.get("host", "")) or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        return Response("This demo accepts local connections only.", status_code=403)
    origin = request.headers.get("origin")
    if (origin and not allowed_origin(origin, request.headers.get("host", ""), request.url.scheme)) or (request.method not in {"GET", "HEAD", "OPTIONS"} and not origin):
        return Response("Origin not allowed", status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


async def discard_session(session: AppraisalSession):
    session.deleted = True
    session.sweeping = False
    sessions.pop(session.session_id, None)
    if session.live_socket:
        with contextlib.suppress(Exception):
            await session.live_socket.close(code=1000)
    for task in list(session.tasks):
        task.cancel()
    if session.tasks:
        await asyncio.gather(*list(session.tasks), return_exceptions=True)
    session.evidence_frames.clear()
    session.last_frame = None


async def cleanup_sessions():
    for session in list(sessions.values()):
        if time.monotonic() - session.updated_at > SESSION_TTL:
            await discard_session(session)


def owned_session(session_id: str, owner: str | None):
    session = sessions.get(session_id)
    if not session or session.deleted or not owner or not secrets.compare_digest(session.owner, owner):
        raise HTTPException(404, "Appraisal not found or expired. Start a new appraisal.")
    if time.monotonic() - session.updated_at > SESSION_TTL:
        raise HTTPException(410, "Appraisal expired. Start a new appraisal.")
    session.updated_at = time.monotonic()
    return session


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: Request, response: Response) -> SessionResponse:
    await cleanup_sessions()
    owner = request.cookies.get("appraisal_owner") or secrets.token_urlsafe(32)
    if len(sessions) >= MAX_SESSIONS or sum(s.owner == owner for s in sessions.values()) >= 4:
        raise HTTPException(429, "Too many appraisals. Close or reset an existing appraisal first.")
    session = AppraisalSession(session_id=uuid.uuid4().hex, owner=owner)
    sessions[session.session_id] = session
    default_locale = [p.strip() for p in os.getenv("APPRAISAL_DEFAULT_LOCALE", "").split(",") if p.strip()]
    if len(default_locale) >= 2 and _has_api_key():
        with contextlib.suppress(Exception):
            await set_pricing_location(session, default_locale[0], default_locale[-1])
    response.set_cookie("appraisal_owner", owner, httponly=True, samesite="strict", max_age=SESSION_TTL, secure=request.url.scheme == "https")
    return SessionResponse(session_id=session.session_id, model=MODEL, has_api_key=_has_api_key(), state=_ui_state(session))


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("appraisal_owner"))
    return {"session_id": session.session_id, "state": _ui_state(session), "has_api_key": _has_api_key()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("appraisal_owner"))
    await discard_session(session)
    return {"deleted": True}


@app.get("/api/sessions/{session_id}/packet")
def download_packet(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("appraisal_owner"))
    inventory = session.store.to_state()
    workflow = session.last_workflow or {}
    room = session.store.room().model_dump()
    markdown = workflow.get("final_markdown") or "# Library Contents Appraisal\n\nRun sync_appraisal_packet to build the full packet."
    if session.floor_plan:
        markdown += f"\n\n## Floor plan\n\n- [Floor plan v{session.floor_plan['version']}](floor_plan.png): illustration of the layout, not a survey drawing. Measurements above are authoritative.\n"
    used = {f for e in [*inventory["books"], *inventory["items"]] for f in e["frame_ids"]}
    used |= {note["frame"] for note in inventory["notes"]}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("appraisal.md", markdown)
        archive.writestr("books.csv", books_csv(inventory))
        archive.writestr("items.csv", items_csv(inventory))
        archive.writestr("room.json", json.dumps(room, indent=2))
        archive.writestr("underwriting.json", json.dumps(underwriting_review(inventory, room), indent=2))
        archive.writestr("inventory.json", json.dumps(inventory, indent=2))
        if session.floor_plan:
            archive.writestr("floor_plan.png", session.floor_plan_image)
        for frame_id in sorted(used & set(session.evidence_frames)):
            archive.writestr(f"evidence/{frame_id}.jpg", session.evidence_frames[frame_id])
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="library-appraisal-{session_id[:8]}.zip"'},
    )


@app.get("/api/sessions/{session_id}/floor-plan")
def floor_plan_image(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("appraisal_owner"))
    if not session.floor_plan:
        raise HTTPException(status_code=404, detail="No floor plan yet")
    return Response(session.floor_plan_image, media_type=session.floor_plan["mime_type"], headers={"Cache-Control": "no-store"})


def set_camera_mode(session: AppraisalSession, enabled: bool) -> bool:
    if session.camera_enabled == enabled:
        return False
    session.camera_enabled = enabled
    if not enabled:
        session.last_frame = None
    return True


@app.on_event("startup")
async def start_cleanup():
    async def loop():
        while True:
            await asyncio.sleep(60)
            await cleanup_sessions()

    app.state.cleanup_task = asyncio.create_task(loop())


@app.on_event("shutdown")
async def stop_cleanup():
    task = getattr(app.state, "cleanup_task", None)
    if task:
        task.cancel()


# ------------------------------------------------------------------ live call

@app.websocket("/ws/live")
async def live_voice(websocket: WebSocket) -> None:
    host = websocket.headers.get("host", "")
    if not local_host(host) or websocket.client.host not in {"127.0.0.1", "::1", "testclient"} or not allowed_origin(websocket.headers.get("origin"), host, "https" if websocket.url.scheme == "wss" else "http"):
        await websocket.close(code=1008)
        return
    try:
        session = owned_session(websocket.query_params.get("session_id", ""), websocket.cookies.get("appraisal_owner"))
    except HTTPException:
        await websocket.close(code=1008)
        return
    if session.live_socket is not None:
        await websocket.close(code=1008)
        return
    session.live_socket = websocket
    await websocket.accept()
    from google.genai import types
    send_lock = asyncio.Lock()
    tasks = set()
    tool_tasks = {}
    pending = {"Claimant": {"id": uuid.uuid4().hex, "text": ""}, "Agent": {"id": uuid.uuid4().hex, "text": ""}}

    async def send(payload):
        if session.deleted:
            return
        async with send_lock:
            await websocket.send_json({**payload, "session_id": session.session_id})

    async def push_state():
        await send({"type": "state", "state": _ui_state(session)})

    session.notify = push_state

    def track(task):
        tasks.add(task)
        return _track(session, task)

    async def finalize(speaker):
        turn = pending[speaker]
        if not turn["text"].strip():
            return
        append_turn(session, speaker, turn["text"], turn["id"])
        await send({"type": "transcript", "speaker": speaker, "text": turn["text"], "id": turn["id"], "final": True})
        pending[speaker] = {"id": uuid.uuid4().hex, "text": ""}

    async def publish_tool(entry):
        session.tool_activity = [item for item in session.tool_activity if item["id"] != entry["id"]] + [dict(entry)]
        session.tool_activity = session.tool_activity[-30:]
        await send({"type": "tool", **entry})

    async def execute_tool(fc, live_session):
        started = time.monotonic()
        name, args, call_id = str(fc.name or ""), dict(fc.args or {}), str(fc.id or uuid.uuid4().hex)
        entry = {"id": call_id, "name": name, "args": args, "phase": "running", "headline": tool_headline(name, args, None)}
        await publish_tool(entry)
        urgent = False
        try:
            if name == "set_pricing_location":
                locale = await set_pricing_location(session, str(args.get("city", "")), str(args.get("country", "")), str(args.get("currency", "")))
                result = {"pricing_location": locale.label, "currency": locale.currency}
            elif name == "start_shelf_sweep":
                set_sweeping(session, True)
                result = {"sweeping": True, "camera_enabled": session.camera_enabled}
                if not session.camera_enabled:
                    result["note"] = "Camera is off. Ask the claimant to turn it on."
            elif name == "stop_shelf_sweep":
                set_sweeping(session, False)
                totals = inventory_totals(session.store.to_state())
                result = {"sweeping": False, "books_found": totals["books"]["count"], "non_book_items": totals["items"]["count"],
                          "frames_scanned": session.store.frames_scanned, "still_pricing": session.store.pending_count()}
            elif name == "pin_item_photo":
                if not session.last_frame or time.monotonic() - session.last_frame_at > FRAME_MAX_AGE_SECONDS:
                    result = {"pinned": False, "message": "No recent camera frame. Ask the claimant to show the item."}
                else:
                    frame_id, frame = session.last_frame_id, session.last_frame
                    session.evidence_frames[frame_id] = frame
                    session.store.notes.append({"frame": frame_id, "note": str(args.get("note", ""))[:500]})
                    result = {"pinned": True, **await _scan_frame(session, frame_id, frame, force=True)}
            elif name == "calibrate_room_scale":
                result = await calibrate_room_scale(session, str(args.get("object_name", "object")), str(args.get("dimension", "height")), float(args.get("real_size_m") or 0))
            elif name == "set_item_count":
                result = await set_item_count(session, str(args.get("item_name", "")), int(args.get("quantity") or 0))
            elif name == "draw_floor_plan":
                result = await draw_floor_plan(session, args)
            elif name == "sync_appraisal_packet":
                await finalize("Claimant")
                result = await sync_appraisal_packet(session)
                urgent = bool(result.get("specialist_referrals"))
            else:
                result = {"error": "Unknown tool"}
        except asyncio.CancelledError:
            entry.update(phase="cancelled", headline="Cancelled")
            with contextlib.suppress(Exception):
                await publish_tool(entry)
            raise
        except Exception:
            logger.exception("Tool %s failed", name)
            result = {"error": f"{name} failed. Continue the conversation and retry if needed."}
        scheduling = scheduling_for(urgent=urgent)
        entry.update(phase="error" if "error" in result else "done", headline=result.get("error") or tool_headline(name, args, result), duration_ms=int((time.monotonic() - started) * 1000), scheduling=scheduling.value)
        await publish_tool(entry)
        await push_state()
        await live_session.send_tool_response(function_responses=[types.FunctionResponse(id=call_id, name=name, response=result, scheduling=scheduling)])

    async def launch_tool(fc, live_session):
        if str(fc.id) in tool_tasks:
            return
        if len(tool_tasks) >= 4:
            await live_session.send_tool_response(function_responses=[types.FunctionResponse(id=fc.id, name=fc.name, response={"error": "The team is busy. Wait for current tools to finish."})])
            return
        task = track(asyncio.create_task(execute_tool(fc, live_session)))
        tool_tasks[str(fc.id)] = task
        task.add_done_callback(lambda done, key=str(fc.id): tool_tasks.pop(key, None))

    try:
        if not _has_api_key():
            await send({"type": "error", "message": "A Google API key is required in the server environment."})
            return
        history = [
            types.Content(
                role="user" if turn["speaker"] == "Claimant" else "model",
                parts=[types.Part(text=turn["text"])],
            )
            for turn in session.transcript
            if turn["speaker"] in {"Claimant", "Agent"}
        ]
        settings = avatar_settings()
        avatar_enabled = avatar_description()["enabled"] and websocket.query_params.get("avatar") != "off"
        avatar_image = avatar_reference() if avatar_enabled and settings["image"] else None
        config = build_live_config(
            camera_enabled=session.camera_enabled,
            avatar_name=settings["name"] if avatar_enabled else "",
            avatar_image=avatar_image,
            avatar_voice=settings["voice"] if avatar_enabled else None,
            seed_history=bool(history),
        )
        async with _live_client(avatar_enabled).aio.live.connect(model=LIVE_MODEL_ID, config=config) as live_session:
            if history:
                # Restore dialogue as context before accepting another turn on reconnect.
                await live_session.send_client_content(turns=history, turn_complete=True)
            await send({"type": "session", "model": LIVE_MODEL_ID, "vision_model": vision_model_label(), "tools": TOOL_NAMES, "avatar": avatar_description(avatar_enabled)})
            await push_state()
            await send({"type": "ready"})
            if not history:
                await live_session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text="(The claimant has joined the call. Greet them.)")]),
                    turn_complete=True,
                )

            async def client_to_gemini():
                windows = {"text": deque(), "audio": deque(), "video": deque(), "camera_state": deque(), "sweep": deque(), "locale": deque()}
                while True:
                    raw = await websocket.receive_text()
                    if len(raw) > MAX_MESSAGE_BYTES:
                        await send({"type": "error", "message": "Input exceeded the message size limit."})
                        continue
                    try:
                        message = json.loads(raw)
                        if not isinstance(message, dict):
                            raise ValueError("Expected a JSON object")
                        kind = message.get("type")
                        if kind == "close":
                            await finalize("Claimant")
                            await finalize("Agent")
                            return
                        if kind not in windows:
                            raise ValueError("Unknown input type")
                        now = time.monotonic()
                        window = windows[kind]
                        period, limit = (60, 20) if kind in {"text", "locale", "sweep"} else (1, 100 if kind == "audio" else 5)
                        while window and now - window[0] >= period:
                            window.popleft()
                        if len(window) >= limit:
                            raise ValueError("Input rate limit reached; pause and try again")
                        window.append(now)
                        session.updated_at = now
                        if kind == "camera_state":
                            enabled = message.get("enabled")
                            if not isinstance(enabled, bool):
                                raise ValueError("Camera state must be true or false")
                            if set_camera_mode(session, enabled):
                                await live_session.send_client_content(
                                    turns=types.Content(role="user", parts=[types.Part(text=camera_mode_instruction(enabled))]),
                                    turn_complete=False,
                                )
                                await push_state()
                        elif kind == "sweep":
                            set_sweeping(session, bool(message.get("enabled")))
                            await push_state()
                        elif kind == "locale":
                            parts = [p.strip() for p in str(message.get("text", "")).split(",") if p.strip()]
                            if not parts:
                                raise ValueError("Enter a location as City, Country")
                            await set_pricing_location(session, parts[0] if len(parts) > 1 else "", parts[-1])
                        elif kind == "text":
                            text = message.get("text", "")
                            if not isinstance(text, str) or not text.strip() or len(text) > 8000:
                                raise ValueError("Text must contain 1–8000 characters")
                            turn_id = message.get("id") or uuid.uuid4().hex
                            if not isinstance(turn_id, str) or len(turn_id) > 100:
                                raise ValueError("Invalid turn identifier")
                            if any(t.get("id") == turn_id for t in session.transcript):
                                continue
                            append_turn(session, "Claimant", text, turn_id)
                            await send({"type": "transcript", "speaker": "Claimant", "text": text, "id": turn_id, "final": True})
                            await live_session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True)
                        else:
                            encoded = message.get("data")
                            if not isinstance(encoded, str):
                                raise ValueError("Missing media data")
                            data = base64.b64decode(encoded, validate=True)
                            if not data or len(data) > (512000 if kind == "video" else 128000):
                                raise ValueError("Invalid media size")
                            if kind == "video":
                                if not data.startswith(b"\xff\xd8\xff"):
                                    raise ValueError("Camera frames must be JPEG images")
                                if set_camera_mode(session, True):
                                    await live_session.send_client_content(
                                        turns=types.Content(role="user", parts=[types.Part(text=camera_mode_instruction(True))]),
                                        turn_complete=False,
                                    )
                                session.last_frame, session.last_frame_at = data, now
                                session.last_frame_id = f"F{uuid.uuid4().hex[:10]}"
                                await live_session.send_realtime_input(video=types.Blob(data=data, mime_type="image/jpeg"))
                            else:
                                if len(data) % 2:
                                    raise ValueError("Audio must be PCM16")
                                await live_session.send_realtime_input(audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000"))
                    except (ValueError, TypeError) as exc:
                        await send({"type": "error", "message": str(exc)})

            async def gemini_to_client():
                while True:
                    async for response in live_session.receive():
                        if response.tool_call and response.tool_call.function_calls:
                            await finalize("Claimant")
                            for fc in response.tool_call.function_calls:
                                await launch_tool(fc, live_session)
                        if response.tool_call_cancellation:
                            for call_id in response.tool_call_cancellation.ids or []:
                                task = tool_tasks.get(str(call_id))
                                if task:
                                    task.cancel()
                        content = response.server_content
                        if not content:
                            continue
                        # Clear queued voice/video before forwarding any more content.
                        if content.interrupted:
                            await send({"type": "interrupted"})
                            await finalize("Agent")
                        for speaker, chunk in (("Claimant", content.input_transcription), ("Agent", content.output_transcription)):
                            if speaker == "Agent" and content.interrupted:
                                continue
                            if chunk and chunk.text:
                                if speaker == "Agent":
                                    await finalize("Claimant")
                                pending[speaker]["text"] += chunk.text
                                await send({"type": "transcript", "speaker": speaker, **pending[speaker], "final": False})
                            if chunk and getattr(chunk, "finished", False):
                                await finalize(speaker)
                        if content.model_turn and not content.interrupted:
                            for part in content.model_turn.parts or []:
                                media = live_media_message(part.inline_data) if part.inline_data else None
                                if media:
                                    # Avatar video also streams while listening;
                                    # idle frames must not split the claimant's turn.
                                    if media["type"] == "audio":
                                        await finalize("Claimant")
                                    await send(media)
                        if getattr(content, "turn_complete", False):
                            await finalize("Claimant")
                            await finalize("Agent")
                            await send({"type": "turn_complete"})

            pair = [track(asyncio.create_task(client_to_gemini())), track(asyncio.create_task(gemini_to_client()))]
            done, _ = await asyncio.wait(pair, timeout=20 * 60, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                await send({"type": "error", "message": "The live connection reached 20 minutes. Reconnect to continue this appraisal."})
            for task in done:
                task.result()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Gemini Live session failed")
        with contextlib.suppress(Exception):
            await send({"type": "error", "message": "Live connection ended. Reconnect to continue this appraisal."})
    finally:
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)
        for item in session.tool_activity:
            if item["phase"] == "running":
                item.update(phase="cancelled", headline="Connection ended")
        session.notify = None
        session.live_socket = None
        set_sweeping(session, False)
        set_camera_mode(session, False)
        session.updated_at = time.monotonic()
        with contextlib.suppress(Exception):
            await websocket.close()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/index.html")
def index_alias():
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/app.js")
def javascript():
    return FileResponse(DEMO_DIR / "app.js", media_type="text/javascript")


@app.get("/avatar.js")
def avatar_javascript():
    return FileResponse(DEMO_DIR / "avatar.js", media_type="text/javascript")


@app.get("/styles.css")
def styles():
    return FileResponse(DEMO_DIR / "styles.css", media_type="text/css")
