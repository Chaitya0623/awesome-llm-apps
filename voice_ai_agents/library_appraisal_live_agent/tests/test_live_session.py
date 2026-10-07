"""Exercise the sweep, valuation, and tool paths of the live server without network or devices."""
import asyncio, functools, io, sys, unittest, warnings
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests.offline  # noqa: F401,E402  (before any app import)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import Image
from agent import run_appraisal_workflow
from schemas import FrameScan, ItemReading, RoomView, SpineReading
from live_demo import server as s
from tests.fake_llm import FakeGemini

SHELF = FrameScan(
    spines=[SpineReading(title="Midnight's Children", author="Rushdie", confidence=0.9),
            SpineReading(title="Gitanjali", author="Tagore", confidence=0.8, format="hardcover")],
    items=[ItemReading(category="shelving", name="teak bookcase")],
)
ROOM = FrameScan(
    items=[ItemReading(category="art", name="framed oil portrait"), ItemReading(category="appliance", name="espresso machine")],
    room=RoomView(visible=True, width_m=4, depth_m=5, ceiling_m=3, cues_used=["door 2.0 m"]),
)


def jpeg(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


class FakeGenai:
    """Vision, currency, and object-size replies routed by request shape."""

    def __init__(self, scans):
        self.scans = list(scans)
        self.calls = []
        self.prompts = []
        self.aio = NS(models=NS(generate_content=self.generate))

    async def generate(self, model, contents, config=None):
        if config is not None and config.response_modalities == ["IMAGE"]:
            self.calls.append("sketch")
            self.prompts.append(contents)
            part = NS(inline_data=NS(data=b"\x89PNGplan", mime_type="image/png"))
            return NS(candidates=[NS(content=NS(parts=[part]))])
        if isinstance(contents, str):
            self.calls.append("currency")
            return NS(text="INR", parsed=None)
        prompt = contents[1]
        if prompt.startswith("Estimate the"):
            self.calls.append("size")
            return NS(text="2.0", parsed=None)
        self.calls.append("scan")
        return NS(text="", parsed=self.scans.pop(0))


class LiveSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.genai = FakeGenai([SHELF, ROOM])
        # I0004 is the framed oil portrait from the ROOM frame; the valuer flags it as original art.
        self.fake_llm = FakeGemini(calls=[], collectible_ids=("I0004",))
        patches = [
            patch.object(s, "_client", lambda: self.genai),
            patch.object(s, "_has_api_key", lambda: True),
            patch.object(s, "run_appraisal_workflow", functools.partial(run_appraisal_workflow, model=self.fake_llm)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.session = s.AppraisalSession("live-test", owner="owner")
        self.pushes = 0

        async def notify():
            self.pushes += 1

        self.session.notify = notify

    async def asyncTearDown(self):
        await s.discard_session(self.session)

    async def test_sweep_scan_value_sync_and_calibrate(self):
        session = self.session
        await s.set_pricing_location(session, "Mumbai", "India")
        self.assertEqual(session.store.locale.currency, "INR")

        await s._scan_frame(session, "F1", jpeg("white"))
        skipped = await s._scan_frame(session, "F2", jpeg("white"))
        self.assertTrue(skipped["skipped"])
        await s._scan_frame(session, "F3", jpeg("black"))
        self.assertEqual(self.genai.calls.count("scan"), 2)
        self.assertEqual((len(session.store.books), len(session.store.items)), (2, 3))
        self.assertIn("F1", session.evidence_frames)

        summary = await s.sync_appraisal_packet(session)
        self.assertEqual(session.store.pending_count(), 0)
        self.assertEqual(summary["books_priced"], 2)
        self.assertEqual(summary["currency"], "INR")
        self.assertEqual(summary["routing"], "specialist_review")  # original oil portrait
        self.assertEqual(summary["room"]["floor_m2"], 20.0)
        self.assertTrue(s._ui_state(session)["packet_markdown"].startswith("# Library Contents Appraisal"))

        session.last_frame, session.last_frame_at = jpeg("gray"), s.time.monotonic()
        calibrated = await s.calibrate_room_scale(session, "door", "height", 2.2)
        self.assertTrue(calibrated["calibrated"])
        self.assertAlmostEqual(calibrated["room"]["floor_m2"], round(20 * 1.1 ** 2, 1))
        self.assertGreater(self.pushes, 3)

    async def test_items_wait_for_a_location_before_valuation(self):
        await s._scan_frame(self.session, "F1", jpeg("white"))
        self.assertIsNone(self.session.valuation_task)
        self.assertEqual(self.fake_llm.calls, [])
        await s.set_pricing_location(self.session, "Mumbai", "India")
        await self.session.valuation_task
        self.assertEqual(self.session.store.pending_count(), 0)
        self.assertEqual(self.fake_llm.calls, ["books", "items"])

    async def test_entries_scanned_during_a_pricing_run_are_priced_next(self):
        session = self.session
        real = s.run_appraisal_workflow
        calls = []

        async def run_while_scanning(inventory, **kwargs):
            calls.append(len(inventory["items"]))
            if len(calls) == 1:  # a frame lands while the first batch is being priced
                session.store.add_item(ItemReading(category="lighting", name="brass floor lamp"), "F9")
            return await real(inventory, **kwargs)

        await s.set_pricing_location(session, "Mumbai", "India", "INR")
        with patch.object(s, "run_appraisal_workflow", run_while_scanning):
            await s._scan_frame(session, "F1", jpeg("white"))
            await session.valuation_task
        self.assertEqual(calls, [1, 2])
        self.assertEqual(session.store.pending_count(), 0)
        lamp = next(i for i in session.store.items.values() if i.name == "brass floor lamp")
        self.assertEqual(lamp.status, "priced")

    async def test_failed_graph_run_marks_batch_failed_and_sync_retries(self):
        await s.set_pricing_location(self.session, "Mumbai", "India", "INR")
        with patch.object(s, "run_appraisal_workflow", side_effect=RuntimeError("quota")):
            await s._scan_frame(self.session, "F1", jpeg("white"))
            with self.assertRaises(RuntimeError):
                await self.session.valuation_task
        self.assertTrue(all(e.status == "failed" for e in self.session.store.entries()))
        summary = await s.sync_appraisal_packet(self.session)
        self.assertEqual(summary["still_unpriced"], 0)

    async def test_sweep_loop_scans_only_fresh_frames(self):
        with patch.object(s, "SWEEP_INTERVAL_SECONDS", 0.01):
            self.session.last_frame, self.session.last_frame_id = jpeg("white"), "F1"
            self.session.last_frame_at = s.time.monotonic()
            s.set_sweeping(self.session, True)
            await asyncio.sleep(0.1)
            s.set_sweeping(self.session, False)
            await asyncio.sleep(0.03)
        self.assertEqual(self.genai.calls.count("scan"), 1)
        self.assertEqual(self.session.store.frames_scanned, 1)

    async def test_floor_plan_needs_a_measured_room_then_reuses_and_redraws(self):
        session = self.session
        args = {"layout_description": "Bookcase along the north wall, portrait by the door", "trigger": "automatic"}
        self.assertFalse((await s.draw_floor_plan(session, args))["sketched"])
        self.assertFalse((await s.draw_floor_plan(session, {**args, "trigger": "correction"}))["sketched"])
        await s._scan_frame(session, "F1", jpeg("white"))
        await s._scan_frame(session, "F2", jpeg("black"))

        drawn = await s.draw_floor_plan(session, args)
        self.assertTrue(drawn["sketched"])
        self.assertIn("4.0 m by 5.0 m", self.genai.prompts[0])
        self.assertIn("Floor 20.0 m²", self.genai.prompts[0])
        self.assertIn("teak bookcase", self.genai.prompts[0])
        self.assertEqual(session.floor_plan_image, b"\x89PNGplan")
        self.assertNotIn("floor_plan_image", s._ui_state(session)["floor_plan"])  # served over HTTP, not pushed

        self.assertTrue((await s.draw_floor_plan(session, args))["reused"])
        corrected = await s.draw_floor_plan(session, {"layout_description": "Bookcase on the east wall", "trigger": "correction"})
        self.assertEqual(corrected["version"], 2)
        self.assertEqual(self.genai.calls.count("sketch"), 2)

    async def test_claimant_count_replaces_the_camera_count(self):
        await s._scan_frame(self.session, "F1", jpeg("white"))
        result = await s.set_item_count(self.session, "teak bookcases", 4)
        self.assertEqual((result["updated"], result["quantity"]), (True, 4))
        missing = await s.set_item_count(self.session, "grand piano", 1)
        self.assertFalse(missing["updated"])
        self.assertIn("teak bookcase", missing["known_items"])


if __name__ == "__main__":
    unittest.main()
