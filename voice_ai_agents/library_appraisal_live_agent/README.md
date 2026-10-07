# Library Appraisal Live Agent

A home-contents insurance agent that appraises a home library in one camera sweep. It talks with the claimant while they film their shelves, reads book spines, and values every book at local prices. It also values the furniture, art, and other things in the room, and measures the room's surface area. Built with Gemini 3.8 Live, with an optional live avatar for synchronized voice and video.

It builds on the architecture of the [Insurance Claim Live Agent Team](../insurance_claim_live_agent_team/): the same live voice and camera transport, non-blocking background tools, avatar player, and hybrid ADK graph, applied to appraising contents instead of reporting a loss.

The agent coaches the sweep and fills in an appraisal ledger as it goes. A background agent team prices the inventory with Google Search for the claimant's city and currency, sizes the room, draws a floor plan, and applies underwriting review rules. It then prepares a downloadable appraisal packet for an underwriter to review.

![Appraisal ledger with sample data from the voice and camera interface](assets/library-appraisal-live-agent-ledger.png)

## Features

* **Live conversation:** speak or type, with live transcripts. The agent coaches you shelf by shelf and follows you if you switch language mid-call.
* **Optional avatar:** synchronized voice and video. The avatar is separate from your camera and never counts as evidence.
* **Spine reading:** a continuous sweep reads titles, authors, and formats from the camera. The same book seen in many frames is counted once.
* **Local valuation:** books and contents are priced in the local currency from current local listings, each with a low–high range and sources. Changing the location re-prices everything.
* **Non-book contents:** bookcases, furniture, portraits and art, lamps, rugs, the coffee machine, and decor, with quantities. The agent confirms how many of each repeated piece of furniture there are.
* **Room measurement:** floor, wall, and total surface area plus linear metres of shelving. Scale comes from objects of known size and is calibrated by one real measurement.
* **Floor plan:** a hand-drawn, top-down plan labelled with the measured dimensions and where the bookcases and other contents stand. Corrections redraw it.
* **Underwriting review:** routes possible collectible books, original art, and antiques to a specialist and flags high-value items to schedule separately. It also lists unclear spines, unconfirmed furniture counts, and gaps in the measurements.
* **Packet download:** ZIP with the Markdown appraisal, book and item CSVs, room and underwriting JSON, the floor plan, and evidence frames.
* **Desktop layout:** call and camera on the left, the ledger in the middle, transcript and checklist on the right. Mobile stacks the sections.

## Setup

Requires Python 3.12 and a Google API key with access to the configured models. An OpenAI API key is optional; see [Running on a free Google key](#running-on-a-free-google-key).

```bash
cd voice_ai_agents/library_appraisal_live_agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set these values in `.env`:

```dotenv
GOOGLE_GENAI_USE_VERTEXAI=False
GOOGLE_API_KEY=your-google-api-key
```

Start the app:

```bash
python -m uvicorn live_demo.server:app --reload --host 127.0.0.1 --port 4178
```

Open [localhost:4178](http://127.0.0.1:4178/).

* **Talk:** start the voice conversation. The agent asks where you live first.
* **Show camera:** share the shelves with the agent. A rear camera is used on phones.
* **Sweep:** start or stop continuous scanning by hand. The agent can also start and stop it.
* **Set prices:** type a `City, Country` to set the pricing location without saying it.
* **Appraisal packet:** preview and download the packet.
* **New:** clear the current appraisal and stop its media and background work.

Appraisal data is stored in memory. Download your packet before restarting or resetting. The app does not send anything to an insurer.

## Running on a free Google key

A free-tier Google key can run the voice call but not the rest of the sweep: it allows about 5 frame reads a minute (the sweep sends about 40), and it has little or no Google Search or image quota, so pricing and the floor plan fail. Each of those steps can move to an OpenAI key independently. The voice call, transcripts, and tools stay on Gemini Live.

```dotenv
OPENAI_API_KEY=your-openai-key
APPRAISAL_VISION_PROVIDER=openai     # spine, item, and room reading; currency and object sizes
APPRAISAL_PRICING_PROVIDER=openai    # local prices with OpenAI web_search
APPRAISAL_SKETCH_PROVIDER=openai     # floor plan with gpt-image-2
APPRAISAL_GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview  # optional: faster first reply
```

* `gemini-3.1-flash-live-preview` started speaking in about 1.3 s in testing, and `gemini-2.5-flash-native-audio-latest` in 7–8 s.
* The live avatar needs a model with video output, such as the default `gemini-3.8-live`.

## Optional live avatar

Requires Google Cloud access to Gemini Live and Application Default Credentials. The Python requirements include avatar support.

```bash
gcloud auth application-default login
```

Add to `.env`, then restart the server:

```dotenv
APPRAISAL_AVATAR_NAME=Kira
APPRAISAL_AVATAR_PROJECT=your-project-id
APPRAISAL_AVATAR_LOCATION=us-central1
APPRAISAL_AVATAR_VOICE=Kore
```

* Keep `GOOGLE_API_KEY` and `GOOGLE_GENAI_USE_VERTEXAI=False` for scanning, valuation, and floor plans. Only the avatar connection uses the Cloud project.
* For a custom portrait, set `APPRAISAL_AVATAR_IMAGE` to a PNG under 5 MB, at least 704 × 1280 pixels. It requires a project allowlisted for custom avatars.
* Leave both `APPRAISAL_AVATAR_NAME` and `APPRAISAL_AVATAR_IMAGE` unset for voice-only mode.
* Playback requires Media Source Extensions and H.264/AAC support. Unsupported browsers use voice mode. If autoplay is blocked, select **Play voice and video**.

## Configuration

Restart the server after changing `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APPRAISAL_GEMINI_LIVE_MODEL` | `gemini-3.8-live` | Live conversation and camera |
| `APPRAISAL_VISION_MODEL` | `gemini-3.8-flash` | Spine, item, and room reading per frame |
| `APPRAISAL_VALUATION_MODEL` | `gemini-3.8-flash` | Normalizer and the two Google Search valuation agents |
| `APPRAISAL_SKETCH_MODEL` | `gemini-3.1-flash-image` | Floor plan drawing |
| `APPRAISAL_VOICE` | `Kore` | Agent voice |
| `APPRAISAL_VISION_PROVIDER` | `gemini` | `openai` reads frames, currencies, and object sizes with OpenAI |
| `APPRAISAL_PRICING_PROVIDER` | `gemini` | `openai` prices with OpenAI web search instead of Google Search grounding |
| `APPRAISAL_SKETCH_PROVIDER` | `gemini` | `openai` draws the floor plan with OpenAI |
| `APPRAISAL_OPENAI_VISION_MODEL` | `gpt-5.4-mini` | OpenAI frame reading |
| `APPRAISAL_OPENAI_MODEL` | `gpt-5.4-mini` | OpenAI pricing |
| `APPRAISAL_OPENAI_SKETCH_MODEL` | `gpt-image-2` | OpenAI floor plan |
| `APPRAISAL_AVATAR_NAME` | Empty | Prebuilt avatar; empty disables it unless an image is set |
| `APPRAISAL_AVATAR_PROJECT` | Empty | Avatar Cloud project |
| `APPRAISAL_AVATAR_LOCATION` | `us-central1` | Avatar API region |
| `APPRAISAL_AVATAR_VOICE` | `Kore` | Avatar voice |
| `APPRAISAL_AVATAR_IMAGE` | Empty | Custom portrait path |
| `APPRAISAL_DEFAULT_LOCALE` | Empty | `City, Country` to price in before the claimant names one |

## Sweep and measurement behavior

| Input | Behavior |
| --- | --- |
| Close-up of a shelf | Read every legible spine, skip unreadable ones, merge repeat sightings; books are never listed as items |
| Frame nearly identical to the last one | Skipped without a model call |
| Wide view of the room | Size width, depth, and ceiling from known objects such as doors, spines, and shelf depth |
| Claimant confirms a real size | Rescale the room from that object (factor clamped to 0.5–2×) |
| Claimant points at an item | Pin the frame with their note and scan it immediately |
| Bookcases or furniture that may repeat | Ask for the total and record it; later frames cannot lower it |
| Room measured | Draw a floor plan with the measured sizes; redraw on correction |
| No pricing location yet | Inventory keeps filling; valuation waits for the location |

* Measurements are medians across frames. Uncertainty starts at ±25%, narrows to about ±10% after calibration, and widens when frames disagree.
* Wall area is gross; windows and doors are not deducted.
* The book count is a lower bound: unreadable spines are not listed.
* Identical furniture in different frames looks like one piece to the camera, so repeated furniture counts come from the claimant. Unconfirmed counts are listed in the packet.
* The floor plan is an illustration. The measurements in the ledger and packet are authoritative.
* Each sync prices up to 20 books and 10 items per graph run, for at most 25 runs (about 500 books). Larger libraries keep the remainder listed as not yet priced.

## Try an appraisal

Use a real bookshelf with the camera on. Or run the agent team on a typed description in ADK Web, using the prompts in `examples.py`:

```bash
cd voice_ai_agents
adk web
```

Select `library_appraisal_live_agent` and paste a prompt such as:

> I live in Mumbai, India and want my home library valued for contents insurance. Books on the main shelf: Midnight's Children by Salman Rushdie (hardcover), The God of Small Things by Arundhati Roy (paperback), and an old hardcover of Gitanjali by Rabindranath Tagore that might be an early edition. There is a teak bookcase about 2 m tall, a framed oil portrait of my grandfather, and a De'Longhi espresso machine with six mugs. The room is about 4 m by 5 m with a 3 m ceiling.

The team should price each entry in INR, measure the room, route the possible early edition (and the oil portrait, if the valuer flags it as original art) to a specialist, and write the packet.

### Underwriting routes

| Route | When |
| --- | --- |
| `specialist_review` | The valuer flagged a book as a possible collectible, or an item as original art or an antique |
| `needs_more_evidence` | Nothing captured yet, more than a quarter unpriced, or the room not measured |
| `standard_contents` | Everything priced and measured |

Any single line worth 10% or more of the total is also listed to be scheduled separately.

## Architecture

| Component | Responsibility |
| --- | --- |
| `agent.py` | ADK graph: normalization, pricing queue, book and contents valuation, room measurement, underwriting review, packet |
| `appraisal_rules.py`, `schemas.py` | Valuation parsing, totals, review rules, packet builders, and structured data |
| `inventory.py` | De-duplicating inventory of books and items |
| `room_measurement.py` | Reference sizes, median aggregation, calibration, uncertainty |
| `live_demo/live_tools.py` | Live configuration, prompts, and tool declarations |
| `live_demo/server.py` | Sessions, WebSocket transport, camera sweep, tools, and downloads |
| `live_demo/app.js` | Conversation, camera input, ledger, and floor plan updates |
| `live_demo/avatar.js` | Synchronized avatar playback, interruption, and fallback (from the insurance claim agent) |
| `live_demo/index.html`, `styles.css` | Responsive interface |

* `gemini-3.8-live` handles conversation, camera input, and optional avatar output. The agent takes one step at a time and waits for the claimant before moving on.
* `gemini-3.8-flash` reads each sweep frame (structured output) and, with Google Search, values books and contents.
* `gemini-3.1-flash-image` draws the floor plan.
* Frame reading, pricing, and the floor plan can each run on OpenAI instead (`gpt-5.4-mini` with structured output or `web_search`, and `gpt-image-2`). Pricing on OpenAI replaces only the two valuation nodes; the rest of the graph is unchanged.
* Background tools: `set_pricing_location`, `start_shelf_sweep`, `stop_shelf_sweep`, `pin_item_photo`, `calibrate_room_scale`, `set_item_count`, `draw_floor_plan`, and `sync_appraisal_packet`.
* Tools run without blocking conversation. Specialist referrals can interrupt; other results arrive when idle.
* The appraisal graph also runs in the background whenever new entries are waiting for a price. It prices up to 20 books and 10 items per run, caches results by inventory revision, and runs again for anything scanned while a batch was being priced.

## Tests

Run from the app directory.

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Tests cover:

* Spine de-duplication, item cleanup, confirmed furniture counts, room measurement and calibration, valuation parsing, and underwriting routes.
* The full ADK graph on the live and ADK Web paths, with Gemini and OpenAI pricing.
* The sweep and valuation pipeline, including entries scanned while a batch is being priced.
* The floor plan, avatar configuration, media routing, sessions, and packet downloads.

Model calls and devices are mocked, and the tests pin every provider to its default whatever your `.env` says. Verify live microphone and camera behavior separately.
