"""Gemini 3.8 Live configuration and background tool contracts for the appraisal agent.

Gemini 3.8 Live runs function calls in the background (behavior NON_BLOCKING)
so the voice agent keeps coaching the sweep while the appraisal team works. The
tools here are the bridge between the live call, the claimant's camera, the
vision scanner, and the ADK appraisal graph. Execution lives in server.py, which
owns the session state.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from google.genai import types

LIVE_MODEL_ID = os.getenv("APPRAISAL_GEMINI_LIVE_MODEL", "gemini-3.8-live")
VISION_MODEL_ID = os.getenv("APPRAISAL_VISION_MODEL", "gemini-3.8-flash")
VOICE_NAME = os.getenv("APPRAISAL_VOICE", "Kore")

TOOL_NAMES = [
    "set_pricing_location",
    "start_shelf_sweep",
    "stop_shelf_sweep",
    "pin_item_photo",
    "calibrate_room_scale",
    "sync_appraisal_packet",
]

SYSTEM_INSTRUCTION = """
You are the live voice agent for a home-contents insurance appraisal team. The claimant is showing
you their home library on camera. In one guided sweep the team builds an insurable inventory: every
readable book valued at local prices, every non-book item (bookcases, furniture, portraits and art,
lamps, rugs, coffee machine, decor) valued too, and the room's surface area.
Speak naturally, warmly, and briefly; this is a voice call. Match the claimant's language.

Flow:
1. Greet briefly and ask which city and country they live in, then call set_pricing_location.
2. Ask them to turn on the camera, call start_shelf_sweep, and coach: shelf by shelf, top to bottom,
   left to right, about half a metre from the spines, slowly, tilting the phone for hard-to-read spines.
   Short coaching only ("a bit closer", "slower here", "great, next shelf"). Do not read titles aloud.
3. Ask for a slow wide pass of the whole room: every wall, the floor and ceiling, corners and doors.
4. Ask for ONE real measurement with the object in frame (a bookcase height, a standard door) and
   call calibrate_room_scale.
5. Call stop_shelf_sweep, then sync_appraisal_packet, and summarise: number of books, the value
   range in local currency, the most valuable items, anything routed to a specialist, and floor and
   wall area with the plus-or-minus percent.

Use pin_item_photo when the claimant points at something specific ("my grandfather painted this").
Quote only numbers returned by tools. Values are replacement-cost estimates grounded in local web
prices, not a certified appraisal. Measurements are camera estimates. You cannot confirm coverage;
the insurer decides. If you notice a hazard (water leak, exposed wiring, fire risk), mention it once.
""".strip()


def camera_mode_instruction(enabled: bool) -> str:
    """App state, kept separate from the claimant transcript."""
    if enabled:
        return (
            "APP CAMERA STATE: ON. You can see the claimant's video. Coach the sweep. "
            "This notice is app state, not a claimant statement."
        )
    return (
        "APP CAMERA STATE: OFF. Ask the claimant to turn on the camera before sweeping. "
        "This notice is app state, not a claimant statement."
    )


def _string_param(description: str) -> types.Schema:
    return types.Schema(type=types.Type.STRING, description=description)


def tool_declarations() -> list[types.Tool]:
    """Return the background tools the voice agent can call."""

    locale = types.FunctionDeclaration(
        name="set_pricing_location",
        description=(
            "Set where the claimant lives so every book and item is valued at local prices in the "
            "local currency. Changing it re-values the whole inventory."
        ),
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "city": _string_param("City, for example Mumbai."),
                "country": _string_param("Country, for example India."),
                "currency": _string_param("ISO 4217 code if the claimant named one, otherwise empty."),
            },
            required=["city", "country"],
        ),
    )
    start = types.FunctionDeclaration(
        name="start_shelf_sweep",
        description=(
            "Start continuously scanning the claimant's camera for book spines, non-book items, and "
            "room dimensions. Returns immediately; the inventory fills in on screen."
        ),
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    )
    stop = types.FunctionDeclaration(
        name="stop_shelf_sweep",
        description="Stop the continuous scan. Returns the counts captured so far.",
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    )
    pin = types.FunctionDeclaration(
        name="pin_item_photo",
        description=(
            "Pin the current camera frame as evidence with the claimant's note and scan it right away. "
            "Use when the claimant points at a specific item or tells you its story or provenance."
        ),
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "note": _string_param(
                    "What the claimant said about it in their words: artist, brand, purchase price, history."
                ),
            },
            required=["note"],
        ),
    )
    calibrate = types.FunctionDeclaration(
        name="calibrate_room_scale",
        description=(
            "Calibrate room measurements with one real dimension the claimant confirmed while the "
            "object is in frame. Returns the corrected floor and wall area."
        ),
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "object_name": _string_param("The object measured, for example 'tall bookcase' or 'door'."),
                "dimension": types.Schema(type=types.Type.STRING, enum=["height", "width", "depth"]),
                "real_size_m": types.Schema(
                    type=types.Type.NUMBER,
                    description="The real size in metres. Convert feet and inches first.",
                ),
            },
            required=["object_name", "dimension", "real_size_m"],
        ),
    )
    sync = types.FunctionDeclaration(
        name="sync_appraisal_packet",
        description=(
            "Send the inventory to the background appraisal team. They value anything still unpriced, "
            "measure the room, apply underwriting review, and prepare the downloadable packet. Returns "
            "totals, routing, specialist referrals, and the room measurement."
        ),
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "reason": _string_param("One short phrase on why you are syncing now, e.g. 'sweep finished'."),
            },
        ),
    )
    return [types.Tool(function_declarations=[locale, start, stop, pin, calibrate, sync])]


def build_live_config(*, camera_enabled: bool = False, seed_history: bool = False) -> types.LiveConnectConfig:
    """Configure voice output with the camera and appraisal tools."""

    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        history_config=types.HistoryConfig(initial_history_in_client_content=True) if seed_history else None,
        system_instruction="\n".join([
            SYSTEM_INSTRUCTION,
            camera_mode_instruction(camera_enabled),
            "Reference clock: " + datetime.now().astimezone().isoformat(),
        ]),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
        ),
        # A sweep streams many frames; slide the context window instead of ending the call.
        context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        tools=tool_declarations(),
    )


def scheduling_for(*, urgent: bool) -> types.FunctionResponseScheduling:
    """Pick how the model should react when a background tool result lands."""

    if urgent:
        return types.FunctionResponseScheduling.INTERRUPT
    return types.FunctionResponseScheduling.WHEN_IDLE


def frame_scan_prompt(known_sizes: str) -> str:
    """Prompt for the vision model that reads one sweep frame."""

    return f"""
You are the vision module of a home-contents insurance appraiser surveying a home library.
Inspect this camera frame and report three things.

1. SPINES: every book spine whose title you can actually read, including vertical or rotated text.
   Correct obvious OCR slips but NEVER invent titles. Skip spines you cannot read and lower confidence
   for partial reads. Classify format from spine height and thickness.
2. ITEMS: every insurable object that is not a book: bookcases and shelving, furniture, framed portraits
   and art, lamps, rugs, coffee machines, mugs, electronics, globes, clocks, plants, decor. Group identical
   small things (e.g. 6 ceramic mugs). Note brand, material, or artist signature when visible.
3. ROOM: if walls, floor, or ceiling are visible, estimate width, depth, and ceiling height in metres using
   objects of known size as rulers, plus the running length of shelf boards visible:
{known_sizes}
   Set visible to false for close-up spine shots where the room cannot be judged.
""".strip()


def object_size_prompt(object_name: str, dimension: str, known_sizes: str) -> str:
    return (
        f"Estimate the {dimension} of the {object_name} in this image, in metres, using the same "
        f"reference sizes you would for a room survey:\n{known_sizes}\nReply with only a number."
    )


def summarize_workflow_for_voice(workflow: dict[str, Any]) -> dict[str, Any]:
    """Compact the ADK graph output into what the voice agent needs to say."""

    packet = workflow["appraisal_packet"]
    review = workflow["underwriting_review"]
    room = workflow["room_measurement"]
    totals = packet["totals"]
    inventory = workflow["inventory"]
    priced = lambda entries: sorted(
        (e for e in entries if e.get("price")),
        key=lambda e: -e["price"]["mid"] * e.get("quantity", 1),
    )
    return {
        "pricing_location": packet["locale"],
        "currency": totals["currency"],
        "books_found": totals["books"]["count"],
        "books_priced": totals["books"]["priced"],
        "non_book_items": totals["items"]["count"],
        "total_value_range": [totals["grand"]["low"], totals["grand"]["high"]],
        "total_value_mid": totals["grand"]["mid"],
        "most_valuable": [
            f"{e.get('title') or e.get('name')} ({e['price']['mid'] * e.get('quantity', 1):,.0f})"
            for e in priced([*inventory.get("books", []), *inventory.get("items", [])])[:3]
        ],
        "routing": review["routing"],
        "routing_reason": review["routing_reason"],
        "specialist_referrals": review["specialist_referrals"][:3],
        "room": {
            "floor_m2": room["floor_m2"],
            "wall_m2": room["wall_m2"],
            "total_surface_m2": room["total_surface_m2"],
            "shelf_linear_m": room["shelf_linear_m"],
            "uncertainty_pct": round(room["range_pct"] * 100),
            "calibrated": room["calibrated"],
        },
        "still_unpriced": len(review["unpriced"]),
        "guardrail": "Do not confirm coverage. Values are estimates, not a certified appraisal.",
    }


def tool_headline(name: str, args: dict[str, Any], result: dict[str, Any] | None) -> str:
    """One-line description of a tool call for the activity feed."""

    if name == "set_pricing_location":
        place = ", ".join(p for p in (args.get("city"), args.get("country")) if p) or "location"
        if result is None:
            return f"Setting prices for {place}"
        return f"Pricing in {result.get('currency', '?')} for {place}"
    if name == "start_shelf_sweep":
        return "Sweep started" if result else "Starting the sweep"
    if name == "stop_shelf_sweep":
        if result is None:
            return "Stopping the sweep"
        return f"Sweep stopped: {result.get('books_found', 0)} books, {result.get('non_book_items', 0)} items"
    if name == "pin_item_photo":
        if result is None:
            return "Pinning the camera frame"
        return "Item pinned with note" if result.get("pinned") else str(result.get("message", "No camera frame"))
    if name == "calibrate_room_scale":
        if result is None:
            return "Calibrating room scale"
        if result.get("calibrated"):
            return f"Calibrated: floor {result['room']['floor_m2']} m²"
        return str(result.get("message", "Calibration failed"))
    if name == "sync_appraisal_packet":
        if result is None:
            return "Appraisal team valuing and measuring"
        return f"{str(result.get('routing', '')).replace('_', ' ')}: packet ready"
    return name


__all__ = [
    "LIVE_MODEL_ID",
    "SYSTEM_INSTRUCTION",
    "TOOL_NAMES",
    "VISION_MODEL_ID",
    "build_live_config",
    "camera_mode_instruction",
    "frame_scan_prompt",
    "object_size_prompt",
    "scheduling_for",
    "summarize_workflow_for_voice",
    "tool_declarations",
    "tool_headline",
]
