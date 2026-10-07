"""Deterministic regressions. All model responses here are controlled fixtures."""
import io, json, sys, unittest, uuid, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests.offline  # noqa: F401,E402  (before any app import)
import appraisal_rules as r
from inventory import InventoryStore
from room_measurement import CALIBRATED_RANGE, UNCALIBRATED_RANGE, calibration_factor, estimate_room
from schemas import ItemReading, Locale, PriceEstimate, RoomView, SpineReading
from live_demo import server as s
from fastapi.testclient import TestClient

ORIGIN = "http://127.0.0.1:4178"


def spine(title, author="", conf=0.9, fmt="paperback"):
    return SpineReading(title=title, author=author, confidence=conf, format=fmt)


def view(w, d, h, shelf=0.0, units=0):
    return RoomView(visible=True, width_m=w, depth_m=d, ceiling_m=h, shelf_linear_m_visible=shelf, shelf_units_visible=units)


def priced(store, entry_id, mid, collectible=False):
    entry = store.books.get(entry_id) or store.items[entry_id]
    entry.price = PriceEstimate(low=mid * 0.5, mid=mid, high=mid * 2, currency=store.locale.currency, collectible=collectible)
    entry.status = "priced"


def library(locale=True):
    store = InventoryStore(Locale(city="Mumbai", country="India", currency="INR") if locale else None)
    store.add_spine(spine("Midnight's Children", "Rushdie", fmt="hardcover"), "F1")
    store.add_spine(spine("The God of Small Things", "Roy"), "F1")
    store.add_item(ItemReading(category="shelving", name="teak bookcase"), "F2")
    store.add_item(ItemReading(category="decor", name="ceramic mugs", quantity=6), "F2")
    store.add_room_view(view(4, 5, 3))
    return store


class InventoryTests(unittest.TestCase):
    def test_same_spine_across_frames_is_merged_keeping_clearest_read(self):
        store = InventoryStore()
        a = store.add_spine(spine("The Name of the Wind", "Rothfuss", 0.7), "F1")
        b = store.add_spine(spine("Name of the Wind", "Patrick Rothfuss", 0.95), "F2")
        self.assertIs(a, b)
        self.assertEqual((len(store.books), a.sightings, a.frame_ids, a.confidence), (1, 2, ["F1", "F2"], 0.95))

    def test_ocr_slip_merges_but_series_titles_and_other_authors_do_not(self):
        store = InventoryStore()
        store.add_spine(spine("Harry Potter and the Prisoner of Azkaban", "Rowling"), "F1")
        store.add_spine(spine("Harry Potter and the Prisoner of Azkahan", "Rowling"), "F2")
        store.add_spine(spine("Harry Potter and the Goblet of Fire", "Rowling"), "F2")
        store.add_spine(spine("Collected Poems", "Larkin"), "F3")
        store.add_spine(spine("Collected Poems", "Plath"), "F3")
        self.assertEqual(len(store.books), 4)

    def test_unreadable_spines_are_dropped(self):
        store = InventoryStore()
        self.assertIsNone(store.add_spine(spine("Smudge", conf=0.2), "F1"))
        self.assertIsNone(store.add_spine(spine("  "), "F1"))
        self.assertEqual(store.books, {})

    def test_item_seen_twice_keeps_max_quantity_not_sum(self):
        store = InventoryStore()
        store.add_item(ItemReading(category="decor", name="ceramic mugs", quantity=4), "F1")
        store.add_item(ItemReading(category="decor", name="ceramic mug", quantity=6), "F2")
        (item,) = store.items.values()
        self.assertEqual(item.quantity, 6)

    def test_identical_furniture_in_separate_frames_needs_a_confirmed_count(self):
        store = InventoryStore()
        for frame in ("F1", "F2", "F3", "F4"):
            store.add_item(ItemReading(category="shelving", name="oak bookcase"), frame)
        (item,) = store.items.values()
        self.assertEqual((item.quantity, item.quantity_confirmed), (1, False))
        self.assertIs(store.set_item_count("the oak bookcases", 4), item)
        store.add_item(ItemReading(category="shelving", name="oak bookcase", quantity=2), "F5")
        self.assertEqual((item.quantity, item.quantity_confirmed), (4, True))  # a later frame cannot undo it
        self.assertIsNone(store.set_item_count("grand piano", 1))
        self.assertIsNone(store.set_item_count("oak bookcase", 0))

    def test_books_listed_as_items_are_dropped_and_bookcases_filed_as_shelving(self):
        from inventory import clean_item_readings
        readings = [ItemReading(category="other", name="book row", quantity=7),
                    ItemReading(category="other", name="stack of paperbacks"),
                    ItemReading(category="furniture", name="built-in bookcases", quantity=3),
                    ItemReading(category="decor", name="brass bookends"),
                    ItemReading(category="appliance", name="espresso machine")]
        cleaned = {r.name: r.category for r in clean_item_readings(readings)}
        self.assertEqual(cleaned, {"built-in bookcases": "shelving", "brass bookends": "decor", "espresso machine": "appliance"})

    def test_count_matches_by_head_noun_only_when_unambiguous(self):
        store = InventoryStore()
        store.add_item(ItemReading(category="shelving", name="three-section wood bookcase"), "F1")
        store.add_item(ItemReading(category="furniture", name="leather armchair"), "F1")
        self.assertEqual(store.set_item_count("teak bookcases", 3).quantity, 3)
        store.add_item(ItemReading(category="furniture", name="wicker chair"), "F2")
        store.add_item(ItemReading(category="furniture", name="dining chair"), "F2")
        self.assertIsNone(store.set_item_count("chairs", 4))  # two different chairs: ask, don't guess

    def test_typed_quantities_count_as_confirmed(self):
        state = r.inventory_from_request({"items": [{"category": "shelving", "name": "teak bookcase", "quantity": 3}]})
        self.assertEqual((state["items"][0]["quantity"], state["items"][0]["quantity_confirmed"]), (3, True))

    def test_state_round_trip_keeps_ids_unique(self):
        store = InventoryStore.from_state(library().to_state())
        new = store.add_spine(spine("Sapiens", "Harari"), "F9")
        self.assertNotIn(new.id, {"B0001", "B0002", "I0003", "I0004"})
        self.assertEqual(len(store.books), 3)

    def test_location_change_resets_valuations_and_stale_batches_are_ignored(self):
        store = library()
        priced(store, "B0001", 900)
        self.assertTrue(store.set_locale(Locale(city="Austin", country="USA", currency="USD")))
        self.assertTrue(all(e.status == "pending" and e.price is None for e in store.entries()))
        store.apply_valuations({"currency": "INR", "books": {"B0001": {"low": 1, "mid": 2, "high": 3, "currency": "INR"}}})
        self.assertIsNone(store.books["B0001"].price)


class MeasurementTests(unittest.TestCase):
    def test_areas_from_median_of_frames(self):
        room = estimate_room([view(4, 5, 2.5), view(4, 5, 2.5), view(9, 9, 9), RoomView(visible=False)])
        self.assertEqual((room.width_m, room.depth_m, room.ceiling_m), (4, 5, 2.5))
        self.assertEqual((room.floor_m2, room.wall_m2, room.total_surface_m2, room.samples), (20.0, 45.0, 85.0, 3))
        self.assertEqual(room.range_pct, UNCALIBRATED_RANGE)

    def test_calibration_scales_area_by_square_and_narrows_range(self):
        room = estimate_room([view(4, 5, 2.5)], factor=1.1, calibration_note="door 2.2 m")
        self.assertAlmostEqual(room.floor_m2, round(20 * 1.21, 1))
        self.assertEqual(room.range_pct, CALIBRATED_RANGE)

    def test_noisy_frames_widen_uncertainty_and_factor_is_clamped(self):
        self.assertGreater(estimate_room([view(3, 5, 2.5), view(5, 5, 2.5)]).range_pct, UNCALIBRATED_RANGE)
        self.assertEqual((calibration_factor(1.0, 10.0), calibration_factor(1.0, 0.1), calibration_factor(0, 2)), (2.0, 0.5, 1.0))

    def test_shelf_length_falls_back_to_book_count(self):
        room = estimate_room([view(0, 0, 0, shelf=1.0)], book_count=300)
        self.assertAlmostEqual(room.shelf_linear_m, round(300 * 0.028 / 0.85, 1))


class ValuationRuleTests(unittest.TestCase):
    def test_no_pricing_queue_without_a_location(self):
        queue = r.select_pricing_queue(library(locale=False).to_state())
        self.assertEqual((queue["books"], queue["items"]), ([], []))

    def test_queue_is_batched(self):
        store = library()
        for n in range(30):
            store.add_spine(spine(uuid.uuid4().hex), "F3")
        queue = r.select_pricing_queue(store.to_state())
        self.assertEqual((len(queue["books"]), len(queue["items"])), (r.MAX_BOOKS_PER_RUN, 2))

    def test_quotes_parse_from_fenced_chatter_and_missing_quotes_fail(self):
        queue = r.select_pricing_queue(library().to_state())
        reply = 'Here:\n```json\n[{"id": "B0002", "low": 500, "mid": 300, "high": 200, "sources": ["https://a.in/x"]}]\n```'
        v = r.apply_valuations(queue, reply, "no prices found", ["https://grounded.example"], [])
        self.assertEqual((v["books"]["B0002"]["low"], v["books"]["B0002"]["high"]), (200, 500))
        self.assertEqual(v["books"]["B0002"]["sources"], ["https://a.in/x"])
        self.assertEqual(sorted(v["failed"]), ["B0001", "I0003", "I0004"])
        self.assertEqual(v["currency"], "INR")

    def test_grounding_urls_back_fill_sources(self):
        queue = r.select_pricing_queue(library().to_state())
        v = r.apply_valuations(queue, '[{"id": "B0001", "low": 1, "mid": 2, "high": 3}]', "[]", ["https://grounded.example"])
        self.assertEqual(v["books"]["B0001"]["sources"], ["https://grounded.example"])

    def test_totals_multiply_item_quantity(self):
        store = library()
        priced(store, "B0001", 400)
        priced(store, "I0004", 150)
        totals = r.inventory_totals(store.to_state())
        self.assertEqual((totals["books"]["mid"], totals["items"]["mid"], totals["grand"]["mid"]), (400, 900, 1300))


class UnderwritingTests(unittest.TestCase):
    def review(self, store):
        state = store.to_state()
        return r.underwriting_review(state, r.measure_room(state))

    def fully_priced(self):
        store = library()
        for entry_id, mid in (("B0001", 900), ("B0002", 400), ("I0003", 30000), ("I0004", 150)):
            priced(store, entry_id, mid)
        return store

    def test_priced_and_measured_library_is_standard_but_flags_big_items(self):
        review = self.review(self.fully_priced())
        self.assertEqual(review["routing"], "standard_contents")
        self.assertTrue(any("teak bookcase" in line for line in review["schedule_separately"]))
        self.assertTrue(any("uncalibrated" in note for note in review["measurement_notes"]))

    def test_collectibles_and_original_art_go_to_specialist(self):
        store = self.fully_priced()
        priced(store, "B0001", 900, collectible=True)
        store.add_item(ItemReading(category="art", name="oil portrait"), "F5")
        priced(store, "I0005", 20000, collectible=True)
        review = self.review(store)
        self.assertEqual(review["routing"], "specialist_review")
        self.assertEqual(len(review["specialist_referrals"]), 2)

    def test_posters_and_prints_are_not_sent_to_a_specialist(self):
        store = self.fully_priced()
        store.add_item(ItemReading(category="art", name="framed movie poster"), "F5")
        priced(store, "I0005", 800)
        self.assertEqual(self.review(store)["routing"], "standard_contents")

    def test_unconfirmed_furniture_counts_are_listed_in_review_and_packet(self):
        store = self.fully_priced()
        review = self.review(store)
        self.assertEqual(review["unconfirmed_counts"], ["teak bookcase: 1 seen on camera, total not confirmed"])
        store.set_item_count("teak bookcase", 2)
        self.assertEqual(self.review(store)["unconfirmed_counts"], [])
        state = store.to_state()
        room = r.measure_room(state)
        markdown = r.build_appraisal_packet(state, room, review)["markdown"]
        self.assertIn("Counts to confirm with the claimant", markdown)

    def test_mostly_unpriced_or_unmeasured_needs_more_evidence(self):
        self.assertEqual(self.review(library())["routing"], "needs_more_evidence")
        store = self.fully_priced()
        store.room_views.clear()
        self.assertEqual(self.review(store)["routing"], "needs_more_evidence")

    def test_packet_markdown_reports_totals_room_and_limits(self):
        store = self.fully_priced()
        store.set_calibration(1.0, "door height confirmed 2.00 m")
        state = store.to_state()
        room = r.measure_room(state)
        packet = r.build_appraisal_packet(state, room, r.underwriting_review(state, room))
        self.assertIn("Mumbai, India (INR)", packet["markdown"])
        self.assertIn("Floor area: 20.0 m²", packet["markdown"])
        self.assertIn("door height confirmed", packet["markdown"])
        self.assertIn("does not confirm coverage", packet["markdown"])
        self.assertIn("Midnight's Children", r.books_csv(state))


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(s.app, base_url=ORIGIN)
        self.addCleanup(s.sessions.clear)

    def test_session_lifecycle_and_packet_download(self):
        created = self.client.post("/api/sessions", headers={"origin": ORIGIN}).json()
        sid = created["session_id"]
        session = s.sessions[sid]
        session.store = library()
        session.evidence_frames["F1"] = b"\xff\xd8\xffjpeg"
        state = self.client.get(f"/api/sessions/{sid}").json()["state"]
        self.assertEqual((len(state["books"]), state["review"]["routing"]), (2, "needs_more_evidence"))
        archive = zipfile.ZipFile(io.BytesIO(self.client.get(f"/api/sessions/{sid}/packet").content))
        self.assertTrue({"appraisal.md", "books.csv", "items.csv", "room.json", "underwriting.json", "evidence/F1.jpg"} <= set(archive.namelist()))
        self.assertEqual(json.loads(archive.read("room.json"))["floor_m2"], 20.0)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/floor-plan").status_code, 404)
        session.floor_plan_image = b"\x89PNGplan"
        session.floor_plan = {"version": 1, "mime_type": "image/png", "layout": "x", "trigger": "automatic", "room": {}}
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/floor-plan").content, b"\x89PNGplan")
        archive = zipfile.ZipFile(io.BytesIO(self.client.get(f"/api/sessions/{sid}/packet").content))
        self.assertEqual(archive.read("floor_plan.png"), b"\x89PNGplan")
        self.assertIn("](floor_plan.png)", archive.read("appraisal.md").decode())
        self.assertEqual(self.client.delete(f"/api/sessions/{sid}", headers={"origin": ORIGIN}).json(), {"deleted": True})
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").status_code, 404)

    def test_sessions_are_owner_scoped_and_local_only(self):
        sid = self.client.post("/api/sessions", headers={"origin": ORIGIN}).json()["session_id"]
        stranger = TestClient(s.app, base_url=ORIGIN)
        self.assertEqual(stranger.get(f"/api/sessions/{sid}").status_code, 404)
        self.assertEqual(TestClient(s.app, base_url="http://example.com").get("/api/health").status_code, 403)
        self.assertEqual(self.client.post("/api/sessions", headers={"origin": "http://evil.example"}).status_code, 403)

    def test_health_lists_tools(self):
        health = self.client.get("/api/health").json()
        self.assertEqual(len(health["tools"]), 8)
        self.assertIn("draw_floor_plan", health["tools"])


if __name__ == "__main__":
    unittest.main()
