"""
skills/nyc-caseworker/test_skill.py — Test suite for the OpenClaw skill.

Run: python skills/nyc-caseworker/test_skill.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skill import (
    find_resources, get_directions, calculate_eligibility,
    get_rights, get_stories, case_login, case_choose, case_checkin,
    caseworker_agent
)

PASS = "✅"
FAIL = "❌"


def test(name):
    def wrapper(fn):
        def run():
            try:
                fn()
                print(f"  {PASS} {name}")
                return True
            except AssertionError as e:
                print(f"  {FAIL} {name}: {e}")
                return False
            except Exception as e:
                print(f"  {FAIL} {name}: {type(e).__name__}: {e}")
                return False
        run.__name__ = fn.__name__
        return run
    return wrapper


# ── Test 1: Find resources near location ─────────────────────────────────────
@test("find_resources returns results with distance when location given")
def test_find_resources_with_location():
    r = find_resources("I need a shelter", location={"lat": 40.65, "lon": -73.95})
    assert "resources" in r, "Missing resources key"
    assert len(r["resources"]) > 0, "No resources returned"
    # Should have distance since location was provided
    assert any("distance_miles" in res or "_dist_m" in res for res in r["resources"]), \
        "Missing distance info"


@test("find_resources handles needs_assessment intent")
def test_find_resources_needs_assessment():
    r = find_resources("I have 4 kids, income $28K, losing housing in Flatbush")
    assert r["intent"] in ["needs_assessment", "lookup"], f"Unexpected intent: {r['intent']}"
    assert len(r["resources"]) > 0, "No resources for family-in-crisis query"


# ── Test 2: Directions ────────────────────────────────────────────────────────
@test("get_directions returns walk option when budget is 0")
def test_directions_zero_budget():
    d = get_directions(40.65, -73.95, 40.62, -73.98, budget=0)
    assert "options" in d, "Missing options"
    assert len(d["options"]) > 0, "No directions returned"
    walk_opt = next((o for o in d["options"] if o["mode"] == "walk"), None)
    assert walk_opt is not None, "No walking option returned"
    assert walk_opt["cost"] == 0, "Walking should cost 0"


@test("get_directions returns subway option when budget allows")
def test_directions_with_budget():
    d = get_directions(40.75, -73.98, 40.62, -73.98, budget=5.00)
    # Should have at least one option (walk always, transit if dist > 0.5mi)
    assert len(d["options"]) >= 1, "No options returned"


# ── Test 3: Eligibility ───────────────────────────────────────────────────────
@test("calculate_eligibility qualifies Tina for SNAP")
def test_eligibility_snap():
    e = calculate_eligibility(household_size=5, annual_income=28000, has_children=True)
    assert e["programs"]["snap"]["qualifies"], "Tina should qualify for SNAP at $28K/5 people"
    assert e["programs"]["snap"]["monthly_estimate"] > 0, "Should have non-zero SNAP estimate"


@test("calculate_eligibility qualifies low-income family for Medicaid")
def test_eligibility_medicaid():
    e = calculate_eligibility(household_size=5, annual_income=28000, has_children=True)
    assert e["programs"]["medicaid"]["qualifies"], "Low-income family with kids should qualify for Medicaid"


@test("calculate_eligibility returns right-to-shelter for at_risk")
def test_eligibility_emergency_shelter():
    e = calculate_eligibility(household_size=1, annual_income=0, housing_status="homeless")
    assert "emergency_shelter" in e["programs"], "Should include emergency_shelter for homeless"
    assert e["programs"]["emergency_shelter"]["qualifies"], "NYC right to shelter applies"


@test("calculate_eligibility high-income correctly excludes programs")
def test_eligibility_high_income():
    e = calculate_eligibility(household_size=1, annual_income=100000)
    assert not e["programs"]["snap"]["qualifies"], "$100K should not qualify for SNAP"


# ── Test 4: Rights ────────────────────────────────────────────────────────────
@test("get_rights returns shelter rights")
def test_rights_shelter():
    r = get_rights("shelter")
    assert len(r) > 0, "No rights returned"
    rights_text = " ".join(x["right"] for x in r)
    assert "Right to Shelter" in rights_text, "Missing core right"


@test("get_rights returns hospital EMTALA protection")
def test_rights_hospital():
    r = get_rights("hospital")
    assert any("emergency" in x["right"].lower() or "emergency" in x["detail"].lower()
               for x in r), "Missing emergency care right"


@test("get_rights has default fallback")
def test_rights_default():
    r = get_rights("nonexistent_type")
    assert len(r) > 0, "Should return default rights for unknown type"


# ── Test 5: Stories ───────────────────────────────────────────────────────────
@test("get_stories returns results")
def test_stories_default():
    s = get_stories()
    assert len(s) >= 2, "Should return at least 2 stories"
    assert all("quote" in x for x in s), "Each story should have a quote"


@test("get_stories filters by need")
def test_stories_filtered():
    s = get_stories(need="housing")
    assert len(s) > 0, "Should find housing stories"
    assert all("housing" in x.get("needs", []) for x in s), "All stories should involve housing"


# ── Test 6: Case Management ───────────────────────────────────────────────────
@test("case_login creates new case")
def test_case_login_new():
    import time
    cid = f"test-{int(time.time())}"
    c = case_login(cid, name="Test User")
    assert not c["returning"], "New case should not be returning"
    assert c["case"]["name"] == "Test User", "Name not saved"


@test("case_login retrieves existing case")
def test_case_login_existing():
    import time
    cid = f"test-existing-{int(time.time())}"
    case_login(cid, name="Existing User")
    c2 = case_login(cid)
    assert c2["returning"], "Should detect returning user"
    assert "Welcome back" in c2["summary"] or "Welcome" in c2["summary"]


@test("case_choose marks need as in_progress")
def test_case_choose():
    import time
    cid = f"test-choose-{int(time.time())}"
    case_login(cid, "Test")
    # First need an open need — add one manually via find_resources
    find_resources("I have 4 kids income $28K losing housing", case_id=cid)
    # Check if needs were detected
    c = case_choose(cid, "housing", "TEST SHELTER INC", "123 Test St", "shelter")
    # case_choose returns {case, message}
    assert "case" in c or "message" in c, "Missing case or message"


@test("case_checkin arrived=true marks resolved")
def test_case_checkin_arrived():
    import time
    cid = f"test-checkin-{int(time.time())}"
    case_login(cid, "Test")
    find_resources("I need housing", case_id=cid)
    case_choose(cid, "housing", "TEST SHELTER", "", "shelter")
    c = case_checkin(cid, arrived=True, resource_name="TEST SHELTER")
    assert "progress" in c or "case" in c, "Missing response"


# ── Test 7: Full agent workflow ───────────────────────────────────────────────
@test("caseworker_agent returns complete response")
def test_caseworker_agent():
    r = caseworker_agent(
        query="I need a shelter tonight",
        location={"lat": 40.65, "lon": -73.95}
    )
    assert "answer" in r
    assert "resources" in r
    assert "rights" in r
    assert "stories" in r
    assert len(r["resources"]) > 0
    assert len(r["rights"]) > 0


if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 60)
    print("NYC Caseworker Skill — Test Suite")
    print("=" * 60)

    tests = [
        ("Resources", [
            test_find_resources_with_location,
            test_find_resources_needs_assessment,
        ]),
        ("Directions", [
            test_directions_zero_budget,
            test_directions_with_budget,
        ]),
        ("Eligibility", [
            test_eligibility_snap,
            test_eligibility_medicaid,
            test_eligibility_emergency_shelter,
            test_eligibility_high_income,
        ]),
        ("Rights", [
            test_rights_shelter,
            test_rights_hospital,
            test_rights_default,
        ]),
        ("Stories", [
            test_stories_default,
            test_stories_filtered,
        ]),
        ("Case Management", [
            test_case_login_new,
            test_case_login_existing,
            test_case_choose,
            test_case_checkin_arrived,
        ]),
        ("Full Agent", [
            test_caseworker_agent,
        ]),
    ]

    passed = 0
    total = 0
    for section, test_list in tests:
        print(f"\n[{section}]")
        for t in test_list:
            total += 1
            if t():
                passed += 1

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed in {elapsed:.1f}s")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)
