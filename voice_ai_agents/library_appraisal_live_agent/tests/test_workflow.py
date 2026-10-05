"""Run the ADK appraisal graph end to end against an offline Gemini stand-in."""
import sys, unittest, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agent import APP_NAME, create_workflow, root_agent, run_appraisal_workflow
from inventory import InventoryStore
from schemas import ItemReading, Locale, RoomView, SpineReading
from tests.fake_llm import FakeGemini


def library(locale=True):
    store = InventoryStore(Locale(city="Lisbon", country="Portugal", currency="EUR") if locale else None)
    store.add_spine(SpineReading(title="The Book of Disquiet", author="Pessoa", confidence=0.9), "F1")
    store.add_spine(SpineReading(title="Blindness", author="Saramago", confidence=0.9), "F1")
    store.add_item(ItemReading(category="furniture", name="leather armchair"), "F2")
    store.add_room_view(RoomView(visible=True, width_m=3, depth_m=4, ceiling_m=2.6))
    return store


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_path_values_measures_reviews_and_packs(self):
        fake = FakeGemini(calls=[], collectible_ids=("B0001",))
        result = await run_appraisal_workflow(library().to_state(), model=fake)
        self.assertEqual(fake.calls, ["books", "items"])
        books = {b["id"]: b for b in result["inventory"]["books"]}
        self.assertEqual(books["B0001"]["status"], "priced")
        self.assertTrue(books["B0001"]["price"]["collectible"])
        self.assertEqual(books["B0002"]["price"]["sources"], ["https://shop.example/listing"])
        self.assertEqual(result["appraisal_packet"]["totals"]["grand"]["mid"], 1900)
        self.assertEqual(result["room_measurement"]["floor_m2"], 12.0)
        self.assertEqual(result["underwriting_review"]["routing"], "specialist_review")
        self.assertIn("Lisbon, Portugal (EUR)", result["final_markdown"])

    async def test_no_location_skips_the_valuation_agents(self):
        fake = FakeGemini(calls=[])
        result = await run_appraisal_workflow(library(locale=False).to_state(), model=fake)
        self.assertEqual(fake.calls, [])
        self.assertTrue(all(b["status"] == "pending" for b in result["inventory"]["books"]))
        self.assertEqual(result["valuations"]["failed"], [])

    async def test_adk_web_path_normalizes_a_typed_description(self):
        fake = FakeGemini(calls=[], request={
            "city": "Austin", "country": "USA", "currency": "USD",
            "books": [{"title": "Dune", "author": "Frank Herbert", "format": "paperback", "confidence": 0.95},
                      {"title": "Neuromancer", "author": "William Gibson", "confidence": 0.95}],
            "items": [{"category": "decor", "name": "vintage globe"}],
            "room": {"visible": True, "width_m": 3.66, "depth_m": 4.27, "ceiling_m": 2.74},
            "room_dimensions_stated": True,
        })
        service = InMemorySessionService()
        await service.create_session(app_name=APP_NAME, user_id="u", session_id="s")
        runner = Runner(app_name=APP_NAME, agent=create_workflow(model=fake), session_service=service)
        message = types.Content(role="user", parts=[types.Part(text="Austin. Dune, Neuromancer, a vintage globe. Room 12 by 14 ft, 9 ft ceiling.")])
        async for _ in runner.run_async(user_id="u", session_id="s", new_message=message):
            pass
        state = (await service.get_session(app_name=APP_NAME, user_id="u", session_id="s")).state
        self.assertEqual(fake.calls, ["normalizer", "books", "items"])
        self.assertEqual(state["appraisal_packet"]["totals"]["books"]["priced"], 2)
        self.assertTrue(state["room_measurement"]["calibrated"])
        self.assertIn("Austin, USA (USD)", state["final_markdown"])

    def test_root_agent_is_exported_for_adk_web(self):
        names = [agent.name for agent in root_agent.sub_agents]
        self.assertEqual(names[:2], ["NormalizeLibraryDescription", "BuildInventory"])
        self.assertEqual(names[-1], "FinalAppraisalPacket")


if __name__ == "__main__":
    unittest.main()
