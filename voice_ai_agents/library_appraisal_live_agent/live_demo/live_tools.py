"""Gemini 3.8 Live configuration and background tool contracts for the appraisal agent.

Gemini 3.8 Live runs function calls in the background (behavior NON_BLOCKING)
so the voice agent keeps coaching the sweep while the appraisal team works. The
tools here are the bridge between the live call, the claimant's camera, the
vision scanner, the floor-plan sketch model, and the ADK appraisal graph. Execution lives in server.py, which
owns the session state.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from google.genai import types

LIVE_MODEL_ID = os.getenv("APPRAISAL_GEMINI_LIVE_MODEL", "gemini-3.8-live")
VISION_MODEL_ID = os.getenv("APPRAISAL_VISION_MODEL", "gemini-3.8-flash")
SKETCH_MODEL_ID = os.getenv("APPRAISAL_SKETCH_MODEL", "gemini-3.1-flash-image")
# "openai" moves frame reading or floor plans to the OpenAI key, e.g. when a free Google key
# allows only a few frame reads a minute or has no image quota.
VISION_PROVIDER = os.getenv("APPRAISAL_VISION_PROVIDER", "gemini").strip().lower()
SKETCH_PROVIDER = os.getenv("APPRAISAL_SKETCH_PROVIDER", "gemini").strip().lower()
OPENAI_VISION_MODEL = os.getenv("APPRAISAL_OPENAI_VISION_MODEL", "gpt-5.4-mini")
OPENAI_SKETCH_MODEL = os.getenv("APPRAISAL_OPENAI_SKETCH_MODEL", "gpt-image-2")
VOICE_NAME = os.getenv("APPRAISAL_VOICE", "Kore")

TOOL_NAMES = [
    "set_pricing_location",
    "start_shelf_sweep",
    "stop_shelf_sweep",
    "pin_item_photo",
    "calibrate_room_scale",
    "set_item_count",
    "draw_floor_plan",
    "sync_appraisal_packet",
]

SYSTEM_INSTRUCTION = """
You are the live voice agent for a home-contents insurance appraisal team. The claimant is showing
you their home library on camera. In one guided sweep the team builds an insurable inventory: every
readable book valued at local prices, every non-book item (bookcases, furniture, portraits and art,
lamps, rugs, coffee machine, decor) valued too, and the room's surface area.
Speak naturally, warmly, and briefly; this is a voice call.

Language: begin in English, then follow the claimant. If they switch language mid-call (for example
to Hindi), switch with them straight away and stay in that language until they change again. Speak
amounts in that language too, but keep tool arguments (titles, item names, places) as written.

Pacing: this is a guided call, one step at a time. Each turn, say at most two short sentences
about the CURRENT step, then stop and listen. Never describe later steps in advance, and never
string several steps or coaching lines together. Move to the next step only when the claimant says
they are done with this one or tool results show it is done.

Steps, in order:
1. Greet in one sentence and ask which city and country they live in. When they answer, call
   set_pricing_location.
2. Ask them to turn on the camera. Once it is on, call start_shelf_sweep and ask them to start at the
   top shelf, about half a metre from the spines, moving slowly left to right. While they sweep, react
   only to what you see right now, one brief remark at a time (for example, asking them to slow down
   or tilt the phone for a spine you cannot read). Do not read titles aloud. Ask whether there are
   more shelves before moving on.
3. Ask for a slow wide pass of the whole room: every wall, floor and ceiling, corners and doors.
4. Ask for ONE real measurement with the object in frame (a bookcase height, a standard door) and
   call calibrate_room_scale with it.
5. Call draw_floor_plan with what you saw of the layout. Say it is an illustration with the measured
   sizes and ask whether the layout looks right. If they correct it, redraw with trigger "correction".
6. Call stop_shelf_sweep, then sync_appraisal_packet, and summarise in a few sentences: number of
   books, the value range in local currency, the most valuable items, anything routed to a
   specialist, and floor and wall area with the plus-or-minus percent.

Counts: the camera cannot tell identical pieces apart across frames, so a room with four matching
bookcases can look like one. When you see bookcases, chairs, or other furniture that may repeat,
ask once how many there are in total and call set_item_count. Also call it whenever the claimant
states a count ("there are three of those").

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
    count = types.FunctionDeclaration(
        name="set_item_count",
        description=(
            "Record the claimant's total count for a non-book item, replacing the camera's count. "
            "Use for furniture that repeats, such as bookcases or chairs."
        ),
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "item_name": _string_param("The item as you would name it, for example 'oak bookcase'."),
                "quantity": types.Schema(type=types.Type.INTEGER, description="Total number in the room."),
            },
            required=["item_name", "quantity"],
        ),
    )
    floor_plan = types.FunctionDeclaration(
        name="draw_floor_plan",
        description=(
            "Draw a hand-sketched top-down floor plan of the room into the ledger, labelled with the "
            "measured dimensions and where the bookcases and other contents stand. Needs a wide pass "
            "of the room first. Works with the camera on or off."
        ),
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "layout_description": _string_param(
                    "Where things are, from what you saw and the claimant said: which wall each bookcase "
                    "stands on, doors and windows, the desk, chairs, rug, portrait, coffee machine. "
                    "Fold in any corrections. Do not include measurements; the team adds them."
                ),
                "trigger": types.Schema(
                    type=types.Type.STRING,
                    enum=["automatic", "explicit_request", "correction"],
                    description=(
                        "automatic after the room is measured; explicit_request when the claimant asks for "
                        "a plan; correction when they correct an existing plan."
                    ),
                ),
            },
            required=["layout_description", "trigger"],
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
    return [types.Tool(function_declarations=[locale, start, stop, pin, calibrate, count, floor_plan, sync])]


def build_live_config(
    *,
    camera_enabled: bool = False,
    avatar_name: str = "",
    avatar_image: bytes | None = None,
    avatar_voice: str | None = None,
    seed_history: bool = False,
) -> types.LiveConnectConfig:
    """Configure voice or avatar output with the same camera and appraisal tools."""

    avatar_config = None
    if avatar_image:
        avatar_config = types.AvatarConfig(customized_avatar=types.CustomizedAvatar(
            image_data=avatar_image, image_mime_type="png",
        ))
    elif avatar_name:
        avatar_config = types.AvatarConfig(avatar_name=avatar_name)

    return types.LiveConnectConfig(
        response_modalities=["VIDEO" if avatar_name or avatar_image else "AUDIO"],
        avatar_config=avatar_config,
        history_config=types.HistoryConfig(initial_history_in_client_content=True) if seed_history else None,
        system_instruction="\n".join([
            SYSTEM_INSTRUCTION,
            camera_mode_instruction(camera_enabled),
            "Reference clock: " + datetime.now().astimezone().isoformat(),
        ]),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=avatar_voice or VOICE_NAME)
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
   Books are never items: do not list books, rows, or stacks of books here; they belong in SPINES only.
   Every bookcase or bookshelf is category "shelving", named like "teak bookcase". List a bookcase only
   when its whole unit is in frame; a close-up of shelf rows adds no bookcase.
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


def floor_plan_prompt(layout_description: str, room: dict[str, Any], items: list[dict[str, Any]]) -> str:
    """Prompt for the image model: the same notebook pen as the ledger, with the measured numbers."""

    contents = ", ".join(
        f"{i['quantity']} x {i['name']}" if i.get("quantity", 1) > 1 else i["name"]
        for i in items if i.get("category") in {"shelving", "furniture", "art", "appliance", "textile", "lighting"}
    ) or "none recorded"
    return (
        "A top-down architectural floor plan, hand-drawn in black ink on cream notebook paper, the kind "
        "an insurance appraiser sketches in a field ledger. Loose confident lines, small handwritten "
        "labels in the same ink, a light brown wash on bookcases only. No people, no perspective, no "
        "shading gradients, no names or addresses. Draw the room as a "
        f"{room['width_m']} m by {room['depth_m']} m rectangle and write these exact labels: "
        f"'{room['width_m']} m' along one side, '{room['depth_m']} m' along the adjacent side, and "
        f"'Floor {room['floor_m2']} m²' in a corner box. Show doors as arcs and windows as double lines. "
        f"Contents to place: {contents}. Layout: {layout_description.strip()}"
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
        "counts_to_confirm": review.get("unconfirmed_counts", [])[:3],
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
    if name == "set_item_count":
        if result is None:
            return f"Recording {args.get('quantity', '?')} x {args.get('item_name', 'item')}"
        return f"Count confirmed: {result['quantity']} x {result['item']}" if result.get("updated") else str(result.get("message", "Item not found"))
    if name == "draw_floor_plan":
        if result is None:
            return "Sketching the floor plan"
        return "Floor plan pinned to the ledger" if result.get("sketched") else str(result.get("message", "Floor plan failed"))
    if name == "sync_appraisal_packet":
        if result is None:
            return "Appraisal team valuing and measuring"
        return f"{str(result.get('routing', '')).replace('_', ' ')}: packet ready"
    return name


__all__ = [
    "LIVE_MODEL_ID",
    "SKETCH_MODEL_ID",
    "SYSTEM_INSTRUCTION",
    "TOOL_NAMES",
    "VISION_MODEL_ID",
    "build_live_config",
    "camera_mode_instruction",
    "floor_plan_prompt",
    "frame_scan_prompt",
    "object_size_prompt",
    "scheduling_for",
    "summarize_workflow_for_voice",
    "tool_declarations",
    "tool_headline",
]
