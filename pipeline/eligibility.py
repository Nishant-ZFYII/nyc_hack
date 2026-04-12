"""
pipeline/eligibility.py — Benefits eligibility screener.

Calculates eligibility for NYC/federal benefits based on household profile.
Uses 2026 Federal Poverty Level (FPL) guidelines.

References:
  - SNAP: 130% FPL for gross income, 100% for net
  - Medicaid NY: 138% FPL (Adult Expansion), higher for kids/pregnant
  - Cash Assistance (NY Safety Net): varies by household
  - WIC: 185% FPL + child under 5 or pregnant
"""
from typing import Any

# 2026 Federal Poverty Level (annual income)
FPL_2026 = {
    1: 15060, 2: 20440, 3: 25820, 4: 31200,
    5: 36580, 6: 41960, 7: 47340, 8: 52720,
}
FPL_ADDITIONAL = 5380  # per person after 8

# SNAP max benefit per household size (2026 NYC estimate)
SNAP_MAX = {
    1: 292, 2: 536, 3: 768, 4: 975,
    5: 1158, 6: 1390, 7: 1536, 8: 1756,
}
SNAP_ADDITIONAL = 220

# Cash Assistance (NY Safety Net) - rough monthly amounts
CASH_ASSISTANCE = {
    1: 181, 2: 285, 3: 394, 4: 503, 5: 612, 6: 721,
}


def get_fpl(household_size: int) -> float:
    """Get annual FPL for a household size."""
    if household_size <= 8:
        return FPL_2026.get(household_size, 15060)
    return FPL_2026[8] + (household_size - 8) * FPL_ADDITIONAL


def calculate_eligibility(
    household_size: int,
    annual_income: float,
    has_children: bool = False,
    has_pregnant: bool = False,
    has_disabled: bool = False,
    has_senior: bool = False,
    is_veteran: bool = False,
    housing_status: str = "",  # "homeless", "at_risk", "stable"
    has_id: bool = True,
    immigration_status: str = "any",  # "citizen", "lpr", "undocumented", "any"
) -> dict:
    """
    Calculate benefits eligibility for a household.

    Returns dict of programs with {qualifies, estimate, notes, documents_needed}.
    """
    fpl = get_fpl(household_size)
    monthly_income = annual_income / 12.0
    pct_fpl = (annual_income / fpl * 100) if fpl else 0

    results = {}

    # ── SNAP (Food Stamps) ────────────────────────────────────────────────────
    snap_max = SNAP_MAX.get(min(household_size, 8), SNAP_MAX[8])
    if household_size > 8:
        snap_max += (household_size - 8) * SNAP_ADDITIONAL
    snap_estimate = 0
    if annual_income <= fpl * 1.3:  # 130% FPL
        # Rough SNAP benefit: max - 30% of net income
        net_monthly = monthly_income * 0.8  # rough net calc
        snap_estimate = max(0, round(snap_max - 0.3 * net_monthly))

    results["snap"] = {
        "program": "SNAP (Food Stamps)",
        "qualifies": annual_income <= fpl * 1.3,
        "monthly_estimate": snap_estimate,
        "notes": (
            f"Income must be ≤ 130% of Federal Poverty Level (${fpl * 1.3:,.0f}/year). "
            f"You're at {pct_fpl:.0f}% of FPL."
        ),
        "documents_needed": ["ID or document request form", "Proof of income", "Social Security numbers"],
        "where_to_apply": "HRA Benefits Access Center",
        "how_long": "30 days (7 days for emergency)",
    }

    # ── Medicaid ──────────────────────────────────────────────────────────────
    medicaid_threshold = 1.38  # 138% FPL for adults
    if has_pregnant or has_children:
        medicaid_threshold = 2.23  # 223% FPL for pregnant / kids under 1
    medicaid_qualifies = annual_income <= fpl * medicaid_threshold

    results["medicaid"] = {
        "program": "Medicaid (Health Insurance)",
        "qualifies": medicaid_qualifies,
        "monthly_estimate": "Free health coverage" if medicaid_qualifies else None,
        "notes": (
            f"Income must be ≤ {int(medicaid_threshold * 100)}% FPL "
            f"(${fpl * medicaid_threshold:,.0f}/year for your household size)."
        ),
        "documents_needed": ["ID or document request form", "Proof of income", "Proof of NY residence"],
        "where_to_apply": "HRA Medicaid Office or NY State of Health",
        "how_long": "45 days (24 hours emergency Medicaid)",
    }

    # ── WIC (Women, Infants, Children) ────────────────────────────────────────
    wic_qualifies = (has_pregnant or has_children) and annual_income <= fpl * 1.85
    results["wic"] = {
        "program": "WIC (Nutrition for families)",
        "qualifies": wic_qualifies,
        "monthly_estimate": "~$55 food vouchers + formula" if wic_qualifies else None,
        "notes": (
            "For pregnant women, new moms, and kids under 5. "
            f"Income ≤ 185% FPL (${fpl * 1.85:,.0f}/year)."
        ),
        "documents_needed": ["ID", "Proof of income", "Proof of residence", "Child's immunization record"],
        "where_to_apply": "Any WIC center (NYC Dept of Health)",
        "how_long": "Same day enrollment possible",
    }

    # ── Cash Assistance (NY Safety Net / TANF) ────────────────────────────────
    cash_max = CASH_ASSISTANCE.get(min(household_size, 6), 721)
    cash_qualifies = annual_income <= fpl * 1.0
    results["cash_assistance"] = {
        "program": "Cash Assistance (NY Safety Net)",
        "qualifies": cash_qualifies,
        "monthly_estimate": cash_max if cash_qualifies else 0,
        "notes": (
            "Monthly cash help for eligible families. "
            "Must meet work requirements or exemptions apply."
        ),
        "documents_needed": ["ID", "Proof of income", "SSN", "Proof of housing costs"],
        "where_to_apply": "HRA Job Center",
        "how_long": "30 days (7 days emergency)",
    }

    # ── School Meals (Free) ───────────────────────────────────────────────────
    results["school_meals"] = {
        "program": "Free School Meals",
        "qualifies": has_children,
        "monthly_estimate": "Free breakfast + lunch for every school day" if has_children else None,
        "notes": "All NYC public school students get free meals regardless of income.",
        "documents_needed": ["School enrollment"],
        "where_to_apply": "Your child's school",
        "how_long": "Immediate",
    }

    # ── Emergency Assistance ──────────────────────────────────────────────────
    if housing_status in ("homeless", "at_risk"):
        results["emergency_shelter"] = {
            "program": "Emergency Shelter (Right to Shelter)",
            "qualifies": True,  # NYC has right-to-shelter law
            "monthly_estimate": "Free shelter for as long as needed",
            "notes": (
                "NYC law: You have a legal RIGHT to shelter. "
                "You cannot be turned away if you're homeless. "
                "Apply at a PATH intake center (families) or a drop-in center (singles)."
            ),
            "documents_needed": ["None required initially — ID helpful but not mandatory"],
            "where_to_apply": "PATH (151 E 151st St, Bronx) for families, Drop-in centers for singles",
            "how_long": "Same day placement guaranteed",
        }

    # ── Free MetroCard (Fair Fares NYC) ────────────────────────────────────────
    results["fair_fares"] = {
        "program": "Fair Fares NYC (50% off MetroCard)",
        "qualifies": annual_income <= fpl * 1.2,
        "monthly_estimate": "~$75/month savings",
        "notes": "Half-price MetroCards for low-income New Yorkers.",
        "documents_needed": ["ID", "Proof of income", "Proof of NYC residence"],
        "where_to_apply": "HRA Benefits Center or fair.fares.nyc",
        "how_long": "7 days",
    }

    # ── Document Assistance (for those without ID) ─────────────────────────────
    if not has_id:
        results["document_assistance"] = {
            "program": "Emergency ID / Documents",
            "qualifies": True,
            "monthly_estimate": "Free replacement IDs",
            "notes": (
                "HRA offers FREE help getting birth certificates, "
                "social security cards, and state IDs. You can apply for benefits "
                "WITHOUT ID using the 'Request for Proof' form."
            ),
            "documents_needed": ["What you have — they'll help with the rest"],
            "where_to_apply": "HRA Benefits Center",
            "how_long": "2-4 weeks for documents, same day for forms",
        }

    # ── Summary totals ────────────────────────────────────────────────────────
    total_monthly = sum(
        r["monthly_estimate"] if isinstance(r.get("monthly_estimate"), (int, float)) else 0
        for r in results.values()
    )
    qualifying = [r["program"] for r in results.values() if r["qualifies"]]

    return {
        "household_size": household_size,
        "annual_income": annual_income,
        "pct_fpl": round(pct_fpl, 0),
        "fpl_for_household": fpl,
        "qualifying_programs": qualifying,
        "estimated_monthly_benefits": total_monthly,
        "programs": results,
    }


# ── Rights database ──────────────────────────────────────────────────────────

RIGHTS = {
    "shelter": [
        {"right": "Right to Shelter", "detail": "NYC legally must provide shelter to anyone who qualifies. You cannot be turned away."},
        {"right": "Families stay together", "detail": "Shelters cannot separate parents from their children."},
        {"right": "Pets are allowed", "detail": "Since 2023, NYC shelters must accommodate pets (some restrictions apply)."},
        {"right": "No ID required initially", "detail": "You can enter shelter without ID. Staff will help you get one."},
        {"right": "Leave anytime", "detail": "You are not required to stay. You can leave for any reason."},
        {"right": "Religious protections", "detail": "Shelters must respect your religious practices (diet, clothing, prayer)."},
        {"right": "Language access", "detail": "You have the right to translation services in 10+ languages."},
    ],
    "food_bank": [
        {"right": "No proof of income needed", "detail": "Most food pantries don't require income verification."},
        {"right": "Culturally appropriate food", "detail": "Large pantries offer halal, kosher, and vegetarian options."},
        {"right": "Emergency food same-day", "detail": "You can get food the same day you arrive, no appointment needed."},
        {"right": "No citizenship check", "detail": "Food pantries do not check immigration status."},
    ],
    "hospital": [
        {"right": "Emergency care cannot be denied", "detail": "EMTALA law: hospitals MUST treat emergencies regardless of ability to pay or immigration status."},
        {"right": "Financial assistance", "detail": "NYC hospitals offer sliding-scale fees based on income."},
        {"right": "Language interpreters", "detail": "Free interpretation in 100+ languages — you can request it."},
        {"right": "Your records are private", "detail": "HIPAA protects your info — it cannot be shared with ICE or other agencies."},
    ],
    "school": [
        {"right": "Free education for all children", "detail": "ALL kids ages 5-21 can attend NYC public school regardless of immigration status or housing situation."},
        {"right": "No address needed", "detail": "Homeless children can enroll WITHOUT a permanent address (McKinney-Vento Act)."},
        {"right": "Free meals", "detail": "Every NYC public school student gets free breakfast and lunch."},
        {"right": "Free MetroCards", "detail": "Students get free MetroCards to travel to school."},
        {"right": "Language support", "detail": "ELL (English Language Learner) programs available."},
    ],
    "benefits_center": [
        {"right": "Apply without ID", "detail": "You can apply for benefits without ID using a 'Request for Proof' form. Staff will help you get ID."},
        {"right": "Emergency SNAP in 7 days", "detail": "If you have < $150/month in cash and rent+utilities > monthly income, you qualify for expedited SNAP in 7 days."},
        {"right": "Language assistance", "detail": "Free translation in person and over the phone."},
        {"right": "Denial appeal", "detail": "If denied, you have 60 days to appeal. Free lawyers can help."},
        {"right": "Confidentiality", "detail": "Your info cannot be shared with ICE or used against you for benefits you already received."},
    ],
    "domestic_violence": [
        {"right": "Confidential location", "detail": "DV shelter addresses are not public for your safety."},
        {"right": "No immigration consequences", "detail": "Immigration status doesn't affect your access to DV services."},
        {"right": "Free legal help", "detail": "Free lawyers for orders of protection, custody, divorce."},
        {"right": "Emergency U-visa", "detail": "DV victims may qualify for U-visa to legalize immigration status."},
    ],
    "default": [
        {"right": "Translation services", "detail": "Free interpretation in 10+ languages at all city services."},
        {"right": "File a complaint", "detail": "If you're treated unfairly, call 311 or contact the NYC Commission on Human Rights."},
        {"right": "Your info is protected", "detail": "NYC agencies cannot share your info with ICE or immigration enforcement."},
    ],
}


def get_rights(resource_type: str) -> list:
    """Get know-your-rights info for a resource type."""
    return RIGHTS.get(resource_type, RIGHTS["default"])


# ── Stories database ──────────────────────────────────────────────────────────

STORIES = [
    {
        "id": "maria-housing",
        "name": "Maria",
        "age_range": "30s",
        "situation": "Single mom, 2 kids, evicted from Brooklyn apartment",
        "needs": ["housing", "benefits", "school"],
        "outcome": "Placed in family shelter in 2 days. Kids continued school. Got SNAP and Medicaid within a week.",
        "timeframe": "1 week from crisis to stable",
        "quote": "I thought I had to figure it out alone. I didn't.",
    },
    {
        "id": "james-reentry",
        "name": "James",
        "age_range": "40s",
        "situation": "Just released from prison, no ID, no money, no family in NYC",
        "needs": ["housing", "documents", "employment"],
        "outcome": "Moved into transitional housing same day. HRA got him new ID in 3 weeks. Started job training program.",
        "timeframe": "30 days to housing + ID + job plan",
        "quote": "I didn't think anyone would help a guy like me. They did.",
    },
    {
        "id": "linh-newcomer",
        "name": "Linh",
        "age_range": "20s",
        "situation": "Recently arrived from Vietnam, limited English, 1 toddler",
        "needs": ["housing", "food", "healthcare", "language"],
        "outcome": "Connected with Asian-American Family Services. Got WIC for toddler. Enrolled in ESL class. Found affordable apartment through HPD.",
        "timeframe": "2 weeks to full safety net",
        "quote": "They spoke Vietnamese. That was everything.",
    },
    {
        "id": "dolores-senior",
        "name": "Dolores",
        "age_range": "70s",
        "situation": "Widow, fixed income, couldn't afford prescription + rent",
        "needs": ["benefits", "healthcare", "housing_assistance"],
        "outcome": "Qualified for SCRIE (Senior rent freeze). Got Medicare Extra Help for prescriptions. Meals on Wheels started weekly deliveries.",
        "timeframe": "3 weeks to all benefits enrolled",
        "quote": "I can pay rent and eat now. I forgot what that felt like.",
    },
    {
        "id": "ahmad-dv",
        "name": "Ahmad",
        "age_range": "30s",
        "situation": "Fleeing domestic abuse with young daughter, undocumented",
        "needs": ["safety", "housing", "legal", "immigration"],
        "outcome": "Moved to confidential DV shelter same night. Got free lawyer for U-visa application and order of protection. Daughter enrolled in school.",
        "timeframe": "24 hours to safety",
        "quote": "Being undocumented didn't matter. Being safe did.",
    },
    {
        "id": "tina-family",
        "name": "Tina",
        "age_range": "40s",
        "situation": "4 kids, lost job, sister kicking her out, diabetic child",
        "needs": ["housing", "healthcare", "benefits", "school"],
        "outcome": "Emergency shelter that night. Pediatric diabetes clinic scheduled for son. SNAP + Medicaid approved in 10 days. Kids enrolled in nearest school.",
        "timeframe": "10 days to full stability",
        "quote": "I was scared, but one step at a time made it possible.",
    },
]


def get_stories(need: str = None, k: int = 3) -> list:
    """Get success stories, optionally filtered by need category."""
    if not need:
        return STORIES[:k]
    # Match stories whose needs include the requested category
    matching = [s for s in STORIES if need in s.get("needs", [])]
    if not matching:
        return STORIES[:k]
    return matching[:k]
