"""
seed_demo_cases.py — Wipe test data and seed curated demo cases.

Usage:
    python3 seed_demo_cases.py               # backup + wipe + seed
    python3 seed_demo_cases.py --no-backup   # skip backup (already backed up)
    python3 seed_demo_cases.py --seed-only   # only seed, don't touch existing

Demo cases cover the full urgency spectrum so the admin dashboard tells a
complete story in one glance.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

CASES_DIR = Path("data/cases")
BACKUP_DIR = Path(f"data/cases_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def iso(days_ago: int = 0, hours_ago: int = 0, minutes_ago: int = 0) -> str:
    dt = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
    return dt.isoformat()


# Borough centroids for realistic lat/lons
BK = {"lat": 40.6782, "lon": -73.9442}  # Brooklyn (Flatbush-ish)
BX = {"lat": 40.8448, "lon": -73.8648}  # Bronx
MN = {"lat": 40.7831, "lon": -73.9712}  # Manhattan
QN = {"lat": 40.7282, "lon": -73.7949}  # Queens
SI = {"lat": 40.5795, "lon": -74.1502}  # Staten Island


# ── Demo cases ───────────────────────────────────────────────────────────────

DEMO_CASES = [
    # ── CRITICAL: DV survivor, fleeing tonight
    {
        "case_id": "NYC-MR2401",
        "name": "Maria R.",
        "created_at": iso(hours_ago=2),
        "last_visit": iso(minutes_ago=15),
        "current_location": BK,
        "needs": [
            {"category": "safety", "priority": 1, "status": "open",
             "notes": "Fleeing domestic violence tonight. Has 6-year-old child.",
             "failed_resources": []},
            {"category": "housing", "priority": 2, "status": "in_progress",
             "notes": "Needs confidential-location DV shelter.",
             "failed_resources": []},
            {"category": "benefits", "priority": 3, "status": "open",
             "notes": "No income documentation, abuser controlled finances.",
             "failed_resources": []},
        ],
        "visits": [
            {"timestamp": iso(minutes_ago=15),
             "query": "Someone broke in, I don't feel safe. I have a 6 year old.",
             "answer": "Connecting you with a confidential shelter."},
        ],
        "destination_intents": [
            {"resource_name": "SAFE HORIZON DV SHELTER (confidential)",
             "resource_address": "CONFIDENTIAL",
             "resource_type": "domestic_violence",
             "state": "intent_confirmed",
             "created_at": iso(minutes_ago=10),
             "need_category": "safety"},
        ],
        "tickets": [
            {"type": "sponsored_ride", "raised_at": iso(minutes_ago=8),
             "status": "open", "reason": "Urgent transport to confidential shelter"},
        ],
        "resources_visited": [],
        "feedback": [],
        "admin_notes": "CRITICAL — DV case, arrange confidential transport ASAP. Do NOT share location in any comms.",
        "emergency_contact": {},
    },

    # ── HIGH: Tina, family of 5, losing housing (THE demo case)
    {
        "case_id": "NYC-TJ8X41",
        "name": "Tina J.",
        "created_at": iso(days_ago=1),
        "last_visit": iso(hours_ago=3),
        "current_location": BK,
        "needs": [
            {"category": "housing", "priority": 1, "status": "open",
             "notes": "Staying with sister in Flatbush, kicked out next week.",
             "failed_resources": []},
            {"category": "benefits", "priority": 2, "status": "in_progress",
             "notes": "Income $28K, 4 kids. Qualifies for SNAP + Medicaid + Fair Fares + Cash Assistance.",
             "failed_resources": []},
            {"category": "school", "priority": 3, "status": "open",
             "notes": "Kids ages 12, 14, 15, 16. Must not interrupt school year (McKinney-Vento).",
             "failed_resources": []},
        ],
        "visits": [
            {"timestamp": iso(days_ago=1),
             "query": "I'm Tina, 4 kids ages 12-16, income $28K, staying with sister in Flatbush but she's kicking me out next week",
             "answer": "Full plan generated — housing + benefits screening + school continuity."},
            {"timestamp": iso(hours_ago=3),
             "query": "Can I still keep my kids in the same school?",
             "answer": "Yes — McKinney-Vento guarantees school continuity even without a permanent address."},
        ],
        "destination_intents": [
            {"resource_name": "HRA BENEFITS ACCESS CENTER — BROOKLYN EAST",
             "resource_address": "404 PINE STREET, BROOKLYN",
             "resource_type": "benefits_center",
             "state": "en_route",
             "created_at": iso(hours_ago=3),
             "need_category": "benefits"},
        ],
        "tickets": [
            {"type": "sponsored_ride", "raised_at": iso(hours_ago=3),
             "status": "open", "reason": "Client requested assistance via NeMo agent"},
        ],
        "resources_visited": [],
        "feedback": [],
        "admin_notes": "",
        "emergency_contact": {},
    },

    # ── HIGH: Reentry, just released from prison
    {
        "case_id": "NYC-JC9K22",
        "name": "James C.",
        "created_at": iso(days_ago=2),
        "last_visit": iso(days_ago=1),
        "current_location": MN,
        "needs": [
            {"category": "housing", "priority": 1, "status": "in_progress",
             "notes": "Just released from prison. No ID, no money, no family in NYC.",
             "failed_resources": []},
            {"category": "benefits", "priority": 2, "status": "open",
             "notes": "Needs replacement ID from HRA. Eligible for cash assistance + SNAP.",
             "failed_resources": []},
            {"category": "employment", "priority": 3, "status": "open",
             "notes": "Interested in job training / reentry program.",
             "failed_resources": []},
        ],
        "visits": [
            {"timestamp": iso(days_ago=2),
             "query": "Just got out, no ID, no money, no family. Need a place to sleep tonight.",
             "answer": "Found reentry transitional housing + HRA for ID + job program."},
        ],
        "destination_intents": [
            {"resource_name": "THE OSBORNE ASSOCIATION, INC",
             "resource_address": "60 CENTRE STREET, MANHATTAN",
             "resource_type": "reentry",
             "state": "arrived",
             "created_at": iso(days_ago=2),
             "need_category": "housing"},
        ],
        "tickets": [],
        "resources_visited": [],
        "feedback": [],
        "admin_notes": "Placed in transitional housing. Still needs ID replacement + job referral.",
        "emergency_contact": {},
    },

    # ── MEDIUM: Family needs SNAP + Medicaid
    {
        "case_id": "NYC-LG4M78",
        "name": "Luis G.",
        "created_at": iso(days_ago=3),
        "last_visit": iso(days_ago=1),
        "current_location": QN,
        "needs": [
            {"category": "benefits", "priority": 1, "status": "in_progress",
             "notes": "Family of 5, income $32K. Eligible for SNAP + Medicaid + WIC (2 kids under 5).",
             "failed_resources": []},
            {"category": "medical", "priority": 2, "status": "open",
             "notes": "Youngest needs immunizations before school enrollment.",
             "failed_resources": []},
        ],
        "visits": [
            {"timestamp": iso(days_ago=3),
             "query": "Family of 5 in Queens, $32K income, need food and healthcare help",
             "answer": "SNAP $687/mo, Medicaid for all 5, WIC for the 2 under 5."},
        ],
        "destination_intents": [
            {"resource_name": "HRA BENEFITS ACCESS CENTER — JAMAICA",
             "resource_address": "165-08 88TH AVENUE, QUEENS",
             "resource_type": "benefits_center",
             "state": "intent_confirmed",
             "created_at": iso(days_ago=1),
             "need_category": "benefits"},
        ],
        "tickets": [],
        "resources_visited": [],
        "feedback": [],
        "admin_notes": "",
        "emergency_contact": {},
    },

    # ── MEDIUM: Elderly, Medicaid + senior services
    {
        "case_id": "NYC-GT5N19",
        "name": "Grace T.",
        "created_at": iso(days_ago=4),
        "last_visit": iso(days_ago=2),
        "current_location": BX,
        "needs": [
            {"category": "medical", "priority": 1, "status": "open",
             "notes": "72yo, diabetic, needs specialist + Medicaid re-enrollment.",
             "failed_resources": []},
            {"category": "benefits", "priority": 2, "status": "in_progress",
             "notes": "Medicare + Medicaid dual eligible. On SNAP already.",
             "failed_resources": []},
        ],
        "visits": [
            {"timestamp": iso(days_ago=4),
             "query": "72 years old, diabetic, Medicaid expired, need help",
             "answer": "Referred to NYC DFTA + HRA for Medicaid re-enrollment."},
        ],
        "destination_intents": [],
        "tickets": [],
        "resources_visited": [],
        "feedback": [],
        "admin_notes": "Coordinate with DFTA case manager.",
        "emergency_contact": {},
    },

    # ── MEDIUM: ESL + employment
    {
        "case_id": "NYC-DP7R05",
        "name": "Dani P.",
        "created_at": iso(days_ago=5),
        "last_visit": iso(days_ago=3),
        "current_location": BK,
        "needs": [
            {"category": "employment", "priority": 1, "status": "open",
             "notes": "Recent arrival, needs ESL + job training. Speaks Korean + basic English.",
             "failed_resources": []},
            {"category": "benefits", "priority": 2, "status": "open",
             "notes": "Undocumented but has kids eligible for SNAP/Medicaid.",
             "failed_resources": []},
        ],
        "visits": [
            {"timestamp": iso(days_ago=5),
             "query": "Need ESL classes and a job, I'm in Brooklyn",
             "answer": "Workforce1 + CUNY ESL + community-based org referrals."},
        ],
        "destination_intents": [],
        "tickets": [],
        "resources_visited": [],
        "feedback": [],
        "admin_notes": "",
        "emergency_contact": {},
    },

    # ── LOW: Resolved/stable — shows the system works
    {
        "case_id": "NYC-SK3V66",
        "name": "Sarah K.",
        "created_at": iso(days_ago=20),
        "last_visit": iso(days_ago=3),
        "current_location": BK,
        "needs": [
            {"category": "housing", "priority": 1, "status": "resolved",
             "notes": "Placed in family shelter 17 days ago. Stable.",
             "failed_resources": []},
            {"category": "benefits", "priority": 2, "status": "resolved",
             "notes": "SNAP + Medicaid approved 14 days ago. $612/mo.",
             "failed_resources": []},
            {"category": "school", "priority": 3, "status": "resolved",
             "notes": "Kids transferred to shelter-zoned school under McKinney-Vento.",
             "failed_resources": []},
        ],
        "visits": [
            {"timestamp": iso(days_ago=20),
             "query": "Single mom, 2 kids, evicted, need shelter tonight",
             "answer": "Placed in family shelter + SNAP/Medicaid pending."},
            {"timestamp": iso(days_ago=3),
             "query": "Everything's going well, thanks for the help",
             "answer": "Glad to hear — case will stay monitored."},
        ],
        "destination_intents": [
            {"resource_name": "FAMILY SHELTER INTAKE — PATH",
             "resource_address": "151 E 151ST STREET, BRONX",
             "resource_type": "shelter",
             "state": "resolved",
             "created_at": iso(days_ago=20),
             "need_category": "housing"},
        ],
        "tickets": [],
        "resources_visited": [],
        "feedback": [
            {"timestamp": iso(days_ago=3), "rating": 5,
             "comment": "Staff at intake were kind. My kids are back in school."},
        ],
        "admin_notes": "Stable placement. Monthly check-in scheduled.",
        "emergency_contact": {},
    },

    # ── LOW: Stable
    {
        "case_id": "NYC-AW2P88",
        "name": "Andre W.",
        "created_at": iso(days_ago=30),
        "last_visit": iso(days_ago=7),
        "current_location": MN,
        "needs": [
            {"category": "employment", "priority": 1, "status": "resolved",
             "notes": "Placed at WorkForce1 job training. Started warehouse role.",
             "failed_resources": []},
            {"category": "benefits", "priority": 2, "status": "resolved",
             "notes": "SNAP approved, tapering as income rises.",
             "failed_resources": []},
        ],
        "visits": [
            {"timestamp": iso(days_ago=30),
             "query": "Need work, don't want to be on benefits forever",
             "answer": "Job training + bridge benefits plan."},
        ],
        "destination_intents": [],
        "tickets": [],
        "resources_visited": [],
        "feedback": [],
        "admin_notes": "Success story — transitioned off emergency benefits within 30 days.",
        "emergency_contact": {},
    },
]


# ── Main ─────────────────────────────────────────────────────────────────────

def backup_existing():
    if not CASES_DIR.exists() or not any(CASES_DIR.iterdir()):
        print("No existing cases — skipping backup.")
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for f in CASES_DIR.glob("*.json"):
        shutil.copy2(f, BACKUP_DIR / f.name)
    print(f"Backed up {len(list(BACKUP_DIR.glob('*.json')))} cases to {BACKUP_DIR}")


def wipe_existing():
    if not CASES_DIR.exists():
        return
    count = 0
    for f in CASES_DIR.glob("*.json"):
        f.unlink()
        count += 1
    print(f"Deleted {count} existing case files")


def seed_demo():
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    for case in DEMO_CASES:
        path = CASES_DIR / f"{case['case_id']}.json"
        with open(path, "w") as f:
            json.dump(case, f, indent=2)
    print(f"Seeded {len(DEMO_CASES)} demo cases into {CASES_DIR}:")
    for case in DEMO_CASES:
        urgency = "CRITICAL" if case["case_id"].startswith("NYC-MR") else (
            "HIGH" if case["case_id"].startswith(("NYC-TJ", "NYC-JC")) else (
                "MEDIUM" if case["case_id"].startswith(("NYC-LG", "NYC-GT", "NYC-DP"))
                else "LOW"))
        open_needs = sum(1 for n in case["needs"] if n["status"] != "resolved")
        print(f"  {case['case_id']:14s}  {case['name']:14s}  "
              f"{urgency:8s}  open_needs={open_needs}  "
              f"tickets={len(case.get('tickets', []))}")


if __name__ == "__main__":
    args = sys.argv[1:]
    seed_only = "--seed-only" in args
    no_backup = "--no-backup" in args

    if not seed_only:
        if not no_backup:
            backup_existing()
        wipe_existing()

    seed_demo()
    print("\nDone. Reload admin portal to see curated cases.")
