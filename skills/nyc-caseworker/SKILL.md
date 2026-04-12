---
name: nyc-caseworker
version: 1.0.0
description: AI caseworker skill for NYC social services. Tracks clients across multiple visits, finds resources by location, handles multi-step journeys (shelter → benefits → healthcare → documents), and adapts recommendations when resources fail. Designed for kiosk deployment at community centers and shelters.
author: NYC Spark Hack Team
license: MIT
capabilities:
  - find_resources_near_location
  - decompose_life_situation
  - track_client_case
  - get_directions_with_budget
  - calculate_benefits_eligibility
  - know_your_rights
  - share_success_stories
  - emergency_shelter_right_to_shelter
permissions:
  - internet:geocoding           # Nominatim API for address lookup
  - internet:routing             # OSRM for walking directions
  - filesystem:read:data         # resource_mart.parquet, graph.pkl
  - filesystem:read:cases        # per-client case files
  - filesystem:write:cases       # save visits + progress
inputs:
  query: string              # natural language situation
  client_id: string          # optional — persists across visits
  location: object           # optional — {lat, lon} for distance sorting
  budget: number             # optional — available money for travel
triggers:
  - "I need shelter"
  - "find help near"
  - "I'm homeless"
  - "apply for benefits"
  - "I was evicted"
  - social services in NYC
---

# NYC Caseworker Skill

## Purpose

This skill turns any AI agent into a real NYC social services caseworker. When deployed at a community center kiosk, it handles the full journey of someone in crisis — from the first query ("I have nowhere to sleep") through benefits enrollment, healthcare, document retrieval, and multi-visit follow-up.

Built on 7,759 verified NYC Open Data resources (shelters, food banks, hospitals, schools, HRA benefits centers, domestic violence services, legal aid) with location-aware search, budget-aware routing, and a persistent case management system.

## When to Use This Skill

Use this skill when a user in NYC needs:
- Emergency shelter, food, or healthcare
- Benefits screening (SNAP, Medicaid, WIC, Cash Assistance, Fair Fares)
- Help navigating complex multi-step journeys (e.g., getting ID → applying for SNAP → finding childcare)
- Location-aware directions with transit/walking/budget options
- Persistent case tracking across multiple visits

## What This Skill Does

### 1. Decomposes a Life Situation into Prioritized Needs

Given "I'm Tina, 4 kids, $28K income, losing housing", the skill identifies:
- Priority 1: Housing (imminent eviction)
- Priority 2: Benefits (SNAP, Medicaid qualify at this income)
- Priority 3: School continuity for 4 kids

It then runs parallel resource searches for each need.

### 2. Location-Aware Resource Matching

When given a location (GPS or typed address), the skill:
- Auto-detects borough from coordinates
- Sorts resources by walking distance
- Shows estimated walk time in minutes
- Filters out resources the client previously marked as full/closed

### 3. Multi-Modal Directions with Budget Awareness

Given the user's current money available:
- Budget $0: suggests walking only, recommends nearest HRA center for free MetroCard
- Budget ≥ $2.90: shows walking vs subway options with travel time
- Uses OSRM (free, open-source) for walking routes
- Uses MTA transit graph for subway estimation

### 4. Persistent Case Management

Across multiple visits, the skill tracks:
- Needs identified (housing, medical, benefits, etc.) with status (open/in_progress/resolved)
- Resources chosen by the client
- Check-ins ("Did you make it to CAMBA?")
- Failed resources (auto-excluded from future queries for this client)
- Progress ("2 of 4 needs resolved")

On return visit, generates a contextual summary: "Welcome back Tina. Housing resolved at CAMBA INC. You still need help with benefits and school."

### 5. Benefits Eligibility Screening

Given household profile, calculates eligibility for:
- **SNAP** (≤130% FPL) with monthly estimate
- **Medicaid** (≤138% FPL adults, ≤223% for kids/pregnant)
- **WIC** (≤185% FPL + kids under 5)
- **Cash Assistance** (NY Safety Net)
- **Fair Fares NYC** (half-price MetroCards)
- **Emergency Shelter** (NYC Right to Shelter law — always qualifies)
- **Free School Meals** (universal for NYC public school students)

Returns estimated monthly amount, documents needed, and where to apply.

### 6. Know Your Rights

For each resource type, explains the legal rights the client has:
- **Shelter**: Right to shelter, pets allowed, no ID required initially, families stay together
- **Food bank**: No income proof needed, no citizenship check
- **Hospital**: Emergency care cannot be denied (EMTALA), free interpreters
- **School**: Free for all kids regardless of immigration/housing, no address required
- **Benefits center**: Apply without ID, language assistance, confidentiality protected from ICE

### 7. Trust via Success Stories

Shows anonymized journeys of people who found help:
- Maria (single mom, 2 kids, evicted — stable in 1 week)
- James (reentry from prison, no ID — housing + ID + job plan in 30 days)
- Linh (newcomer from Vietnam, limited English — full safety net in 2 weeks)
- Dolores (senior on fixed income — all benefits enrolled in 3 weeks)
- Ahmad (DV survivor, undocumented — safety in 24 hours)
- Tina (4 kids, eviction, diabetic child — full stability in 10 days)

## How to Invoke This Skill

### Via API (if running as a service)

```bash
# Find resources near a location
curl -X POST http://localhost:9000/api/query -d '{
  "query": "I need a shelter tonight",
  "case_id": "tina-2024",
  "location": {"lat": 40.653, "lon": -73.950}
}'

# Check benefits eligibility
curl -X POST http://localhost:9000/api/eligibility -d '{
  "household_size": 5,
  "annual_income": 28000,
  "has_children": true,
  "housing_status": "at_risk"
}'

# Get multi-modal directions
curl -X POST http://localhost:9000/api/directions -d '{
  "from_lat": 40.75, "from_lon": -73.98,
  "to_lat": 40.62, "to_lon": -73.98,
  "budget": 0
}'
```

### Via Agent Integration

The skill exposes these tools to the agent:
- `find_resources(query, location, case_id)` → list of matching resources sorted by distance
- `get_directions(from, to, budget)` → walk/transit options with times and costs
- `calculate_eligibility(household_profile)` → qualifying benefit programs
- `get_rights(resource_type)` → legal rights at that resource type
- `get_stories(need_category)` → 2-3 similar success stories
- `case_login(client_id)` → returns progress summary for returning clients
- `case_checkin(client_id, resource, arrived)` → updates client's journey

## Data Sources (all verified, open, NYC government)

- **PLUTO** (MapPLUTO) — 870K NYC tax lots (base spatial layer, overflow site discovery)
- **DOHMH Facilities** — hospitals, clinics, shelters, schools, food banks, senior services
- **NYPD Complaints** — safety scoring (crime density within 500m of each resource)
- **311 Service Requests** — quality scoring (311 complaints within 500m)
- **MTA GTFS** — 496 subway stations with line info
- **HRA Benefits Centers** — SNAP/Medicaid/Cash enrollment offices
- **NYC Domestic Violence Resources** — DV shelters and services
- **DOE Schools** — 2,170 schools

All data from [NYC Open Data](https://data.cityofnewyork.us/) via Socrata API. No scraping. No paid APIs. No PII exposure.

## Architecture: Bounded DSL (Anti-Hallucination)

The LLM **never** answers questions directly or touches the data. Instead:

1. **LLM → JSON Plan** — translates natural language to a structured query plan
2. **Executor → Real Data** — runs the plan against verified data (cuDF on GPU / pandas on CPU)
3. **Synthesizer → Natural Language** — formats raw results into caseworker-style prose
4. **Verifier → Per-Claim Fact Check** — validates every named resource/address against the mart

This means the LLM cannot invent resource names, addresses, or services. Every fact in the answer traces back to a real row in the NYC Open Data databases.

## Why This Matters

NYC has **33,000 DHS caseworkers** and approximately **150,000 people** in shelters on any given night. A caseworker spends ~45 minutes per client doing needs assessment; this skill does it in under 10 seconds with complete resource coverage.

The skill is designed for **kiosk deployment** at shelters, community centers, and libraries — places where people in crisis actually go. Clients interact via touch buttons and minimal typing. Results are printable for offline use.

## Extending to Other Cities

The architecture is city-agnostic. To adapt to another city:
1. Replace the Socrata data sources in `pull_all.py`
2. Update the geocoding landmarks in `pipeline/geocode.py`
3. Update the Federal/state benefits rules in `pipeline/eligibility.py`
4. Rebuild the resource mart and knowledge graph

The NYC version includes optional GPU acceleration via NVIDIA RAPIDS (cuDF, cuGraph, cuOpt, cuML) for sub-second queries at city scale.

## Acknowledgments

Built for **NVIDIA Spark Hack NYC 2026** on the Acer Veriton GN100 DGX Spark with Nemotron-3-Nano LLM.

Inspired by the daily work of NYC DHS, DSS, HRA, and countless nonprofit caseworkers who keep this city's safety net running.
