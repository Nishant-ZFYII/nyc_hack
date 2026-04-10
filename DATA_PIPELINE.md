# Data Pipeline & Technical Documentation

This document explains how the NYC Social Services Intelligence Engine works end-to-end: what data we use, how we clean it, how we build the knowledge graph, and how queries flow through the system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Sources](#data-sources)
3. [Pipeline Steps](#pipeline-steps)
4. [Data Cleaning Details](#data-cleaning-details)
5. [Resource Mart](#resource-mart)
6. [Knowledge Graph](#knowledge-graph)
7. [SPO Triples & txt2kg](#spo-triples--txt2kg)
8. [Query Flow](#query-flow)
9. [LLM Integration](#llm-integration)
10. [Verification & Confidence](#verification--confidence)
11. [User Feedback Loop](#user-feedback-loop)
12. [NVIDIA Stack (DGX Spark)](#nvidia-stack-dgx-spark)

---

## System Overview

The system is an **AI caseworker** for NYC Department of Social Services. It doesn't just find resources — it decomposes life situations into comprehensive needs assessments, screens for benefits eligibility, identifies needs the person didn't know to ask for, and provides explainable reasoning with verification.

**Core architecture: Bounded DSL (Domain-Specific Language)**

```
User query (natural language)
    ↓
LLM Planner (Nemotron on DGX / Claude Haiku locally)
    ↓
Structured JSON Plan (NOT a free-text answer)
    ↓
Executor (queries real data — pandas/cuDF)
    ↓
Synthesizer (formats results into natural language)
    ↓
Verifier (fact-checks each claim against raw data)
    ↓
User sees answer with verification badges
    ↓
User can report ground-truth issues → system adapts
```

**Why this matters:** The LLM never answers directly. It only produces a structured JSON plan. The executor queries real data. This prevents hallucination — every fact in the answer traces back to a real row in the database.

---

## Data Sources

All data comes from **NYC Open Data** (data.cityofnewyork.us) via the Socrata API. No data is hardcoded or fabricated.

| Dataset | Socrata ID | Rows | What it contains |
|---------|-----------|------|------------------|
| **PLUTO** (tax lots) | `64uk-42ks` | 857,161 | Every tax lot in NYC — address, land use, building info. Land use "08" = assembly/community buildings = potential emergency overflow sites |
| **DOHMH Facilities** | `ji82-xba5` | 9,783 | All city-licensed facilities — hospitals, clinics, shelters, schools, senior centers, childcare, mental health, etc. This is the backbone dataset |
| **NYPD Complaints** | `qgea-i56i` | 499,987 | Crime complaints with lat/lon. Used to compute safety scores (fewer crimes nearby = safer) |
| **311 Complaints** | `erm2-nwe9` | 299,988 | HPD housing complaints. Used for service quality scores and txt2kg triple extraction |
| **MTA Subway Stations** | CSV | 496 | All subway stations with lat/lon, subway lines, ADA accessibility |
| **Domestic Violence** | `5ziv-wcy4` | 252 | DV shelters and service providers |
| **Hospitals** | Manual CSV | 78 | NYC Health + Hospitals system facilities with POINT geometry |
| **HRA Benefits Centers** | `9d9t-bmk7` | 29 | SNAP/Medicaid/cash assistance enrollment offices |
| **Homeless Drop-in Centers** | `bmxf-3rd4` | 8 | Overnight drop-in centers |
| **Cooling Centers** | `h2bn-gu9k` | ~200 | Summer cooling / winter warming centers |
| **NYCHA Developments** | `phvi-damg` | ~200 | Public housing developments with MultiPolygon geometry |
| **DOE Schools** | via DOHMH | ~2,170 | Public schools (extracted from DOHMH facility database) |
| **Women's Resources** | `pqg4-dm6b` | ~1,000 | Legal aid, DV services, language services |

### How we pull data (`pull_all.py`)

```python
# Uses sodapy (Socrata API client) to download each dataset
# Pulls in chunks of 50,000 rows with pagination
# Saves raw data as Parquet files in raw/
# Supports optional NYC_APP_TOKEN for higher rate limits
# Skips datasets that already exist (idempotent)
```

Run: `python pull_all.py` (~5-10 minutes)

---

## Pipeline Steps

```
Step 1: pull_all.py      → raw/*.parquet        (download from NYC Open Data)
Step 2: clean_all.py     → stage/*.parquet      (clean, normalize, geocode)
Step 3: build_mart.py    → data/resource_mart.parquet + data/pluto_layer.parquet
Step 4: build_graph.py   → data/graph.pkl       (knowledge graph)
Step 5: build_triples.py → data/triples.parquet + data/triples_311.parquet (optional)
```

Each step is independent and can be re-run. The app only needs `data/resource_mart.parquet` and `data/graph.pkl`.

---

## Data Cleaning Details

### The problem

Raw NYC Open Data is messy:
- Different datasets use different column names (`facname` vs `name` vs `site_name`)
- Coordinates come in different formats (decimal degrees, POINT WKT geometry, State Plane coordinates, MultiPolygon dicts)
- Borough names are inconsistent ("Manhattan" vs "MN" vs "1" vs "New York")
- Resource types aren't labeled — DOHMH has 9,783 facilities spanning hospitals, schools, shelters, food banks, senior centers, childcare, etc. all in one table

### How we clean (`clean_all.py`)

14 cleaning functions, one per dataset type:

#### 1. PLUTO (857K lots)
- Extract: `bbl, address, borough, landuse, latitude, longitude, numfloors, yearbuilt, ownername`
- Normalize borough names to 2-letter codes (MN/BK/QN/BX/SI)
- Zero-pad land use codes (`"8"` → `"08"`)
- Flag overflow candidates: land use 08 (civic/community), 09 (transportation), 12 (open space) — these are buildings that could serve as emergency shelters
- Drop rows outside NYC bounding box (lat 40.4-40.95, lon -74.3 to -73.6)

#### 2. Shelters (from DOHMH)
- DHS daily census is system-wide aggregate (1 row/day), NOT per-shelter — useless for individual shelter locations
- Instead: filter DOHMH facilities where `facgroup` is "HUMAN SERVICES" or `factype` contains shelter/homeless/transitional/drop-in/safe haven/crisis keywords
- Extract lat/lon, normalize borough, assign `resource_type = "shelter"`

#### 3. Hospitals (manual CSV)
- Source has POINT geometry column: `POINT (-73.98 40.75)`
- Parse with regex to extract lon/lat (note: POINT format is lon first, lat second)
- Classify: "hospital" if type contains hospital/acute/trauma, else "clinic"

#### 4. DOHMH Facilities (9,783 rows → 19 resource types)
- This is the most complex cleaning step
- **Classification function** maps each facility to a resource type based on `facgroup`, `factype`, `facsubgrp`:
  - Food pantry/soup kitchen/feeding → `food_bank`
  - Hospital/diagnostic/treatment center/health center → `hospital`
  - Mental health/psychiatric/substance → `mental_health`
  - Shelter/transitional/safe haven/drop-in → `shelter`
  - Elementary/high/middle school → `school`
  - Senior center/elderly/meals on wheels → `senior_services`
  - Day care/early education/head start → `childcare`
  - Legal/law/court → `legal_aid`
  - Youth services → `youth_services`
  - Libraries → `library`
  - Emergency services → `emergency_services`
  - Everything else → `community_center`
- Filter out junk: transportation, solid waste, historical sites, parking, tow trucks, textiles, compost

#### 5. Food Banks (from DOHMH)
- The HRA resources dataset (`7btz-mnc8`) turned out to be a healthcare provider directory, NOT food banks
- Instead: filter DOHMH facilities where `factype` contains food pantry/soup kitchen/feeding site/food bank/emergency food/congregate meals
- This gives us real food bank locations with coordinates

#### 6. MTA Subway Stations
- Source: CSV with `GTFS Latitude`, `GTFS Longitude`, `Daytime Routes`, `ADA` columns
- Rename to standard schema
- `ada_accessible` converted from 0/1 to boolean
- `subway_lines` kept as space-separated string (e.g., "A C E")

#### 7. Schools
- Two sources merged: DOHMH facilities (type="school") + Women's Resources (category contains school/education/literacy/GED)
- Deduplicated by (name, latitude)

#### 8. Domestic Violence Resources
- Lat/lon sometimes in standard columns, sometimes in POINT geometry — try both
- Safety-sensitive: these locations are real, but the system is careful about how it presents them

#### 9. NYCHA Developments
- Raw data has `the_geom` as a Python dict with nested `coordinates` arrays (MultiPolygon)
- No simple lat/lon columns
- **Fix:** Iterate through all coordinate rings in the MultiPolygon, average all points to get centroid
- This gives us the center point of each housing development

#### 10. Cooling Centers
- Raw data uses `x` and `y` columns — these are already decimal degrees (NOT State Plane as initially suspected)
- `x` = longitude (~-74), `y` = latitude (~40)

#### 11. NYPD Complaints (safety signal, not a resource)
- 500K crime complaints with lat/lon
- Keep: latitude, longitude, borough, offense description, complaint date
- Used in build_mart.py to compute safety scores per resource

#### 12. 311 Complaints (quality signal, not a resource)
- 300K HPD housing complaints
- Keep: latitude, longitude, borough, complaint type, descriptor, incident address, date
- Used for quality scores AND txt2kg triple extraction

---

## Resource Mart

**File:** `data/resource_mart.parquet` (1.4 MB, ~7,759 rows)

The resource mart is the unified database of every social service resource in NYC. Built by `build_mart.py` in 5 steps:

### Step 1: Combine all stage files
Loads 11 stage parquet files, ensures each has core columns (`resource_id, resource_type, name, address, borough, latitude, longitude`), concatenates into one DataFrame.

### Step 2: Normalize and validate
- Coerce lat/lon to numeric, drop rows with missing/invalid coordinates
- Spatial dedup: round lat/lon to 4 decimal places, drop duplicates within same resource type
- Re-issue sequential resource IDs (e.g., `shelter_0`, `shelter_1`, `food_ba_0`)
- Fill missing names with "Unknown"

### Step 3: Nearest transit station
For each resource, find the closest MTA subway station using KDTree spatial index:
- Convert lat/lon to meters (NYC: 1° lat ≈ 111,320m, 1° lon ≈ 85,390m)
- Compute walk time: distance / 80 m/min
- Adds columns: `nearest_transit_id`, `nearest_transit_name`, `nearest_transit_dist_m`, `nearest_transit_walk_min`

### Step 4: Safety score from NYPD
For each resource, count NYPD complaints within 500m radius:
- Uses bounding box pre-filter for speed, then exact distance check
- `safety_score = 1 - (count / 99th_percentile)`, clipped to [0, 1]
- Higher score = fewer crimes nearby = safer

### Step 5: Quality score from 311
Same approach as safety, but with 311 HPD complaints:
- `quality_score = 1 - (count / 99th_percentile)`, clipped to [0, 1]
- Higher score = fewer complaints nearby = better service quality

### Final schema

| Column | Type | Description |
|--------|------|-------------|
| resource_id | str | Unique ID (e.g., `shelter_42`) |
| resource_type | str | One of 19 types (see below) |
| name | str | Facility name |
| address | str | Street address |
| borough | str | MN/BK/QN/BX/SI |
| latitude | float | WGS84 latitude |
| longitude | float | WGS84 longitude |
| capacity | int | Bed count (shelters) or capacity |
| ada_accessible | bool | Wheelchair accessible |
| languages_spoken | str | Languages available |
| nearest_transit_name | str | Closest subway station |
| nearest_transit_dist_m | float | Distance to nearest subway (meters) |
| nearest_transit_walk_min | float | Walk time to nearest subway |
| safety_score | float | 0-1, higher = safer (from NYPD data) |
| quality_score | float | 0-1, higher = better (from 311 data) |
| n_nypd_500m | int | NYPD complaints within 500m |
| n_311_500m | int | 311 complaints within 500m |

### Resource type breakdown

| Type | Count | Source |
|------|-------|--------|
| school | ~2,170 | DOHMH + DOE |
| community_center | ~1,800 | DOHMH (catch-all) |
| childcare | ~1,200 | DOHMH |
| hospital | ~800 | DOHMH + manual CSV |
| transit_station | ~496 | MTA CSV |
| senior_services | ~400 | DOHMH |
| shelter | ~350 | DOHMH (human services) |
| mental_health | ~250 | DOHMH |
| food_bank | ~200 | DOHMH |
| domestic_violence | ~252 | Socrata |
| nycha | ~200 | Socrata (public housing) |
| cooling_center | ~200 | Socrata |
| library | ~200 | DOHMH |
| education | ~150 | DOHMH (higher ed) |
| youth_services | ~100 | DOHMH |
| emergency_services | ~50 | DOHMH |
| legal_aid | ~40 | DOHMH |
| benefits_center | ~29 | HRA |
| dropin_center | ~8 | Socrata |

### PLUTO Layer

**File:** `data/pluto_layer.parquet` (38 MB, 857K rows)

Second-tier mart — not loaded at startup, only queried during simulation (e.g., cold emergency scenario needs overflow sites). Contains every tax lot in NYC with:
- `is_overflow_candidate`: True for land use 08 (civic/community buildings like churches, community centers) that could serve as emergency shelters

---

## Knowledge Graph

**File:** `data/graph.pkl` (12 MB)

Built by `build_graph.py`. A directed graph with 3 node types and 5 edge types.

### Node types

| Type | ID range | Count | What |
|------|----------|-------|------|
| Resource | 0 – 99,999 | ~7,200 | Shelters, food banks, hospitals, etc. (excluding transit) |
| Transit station | 100,000 – 100,999 | ~496 | MTA subway stations |
| Census tract | 200,000 – 202,999 | ~800 | Synthetic ~1km grid cells (real ACS tracts on DGX) |

### Edge types

| Edge | Direction | Weight | Count | How it's built |
|------|-----------|--------|-------|----------------|
| **NEAR** | resource ↔ resource | distance (meters) | ~65K | KDTree k=5 nearest neighbors within 2km |
| **WALK_TO_TRANSIT** | resource ↔ station | walk time (min) | ~14K | Each resource to its nearest subway station, distance/80 m/min |
| **TRANSIT_LINK** | station ↔ station | travel time (min) | ~3K | Stations sharing a subway line, ~3 min between stops |
| **IN_TRACT** | resource → tract | 1.0 | ~7K | Each resource to its containing census tract |
| **SERVED_BY** | tract → resource | 1/transit_time | ~400K+ | Transit accessibility: walk to station + subway + walk from station. Key edge for "which neighborhoods can reach which resources" |

### How SERVED_BY edges are built

This is the most expensive computation and the key "why DGX" story:

```
For each census tract:
  1. Find nearest subway station (walk time = distance / 80 m/min)
  2. For each resource:
     a. Estimate subway time = straight-line distance / 500 m/min
     b. Add resource's walk time to its nearest station
     c. Total = walk_to_station + subway + walk_from_station
  3. If total < 60 min, add SERVED_BY edge with weight = 1/total
```

On DGX with cuGraph: real SSSP (shortest path) through the full MTA transit graph.
Locally with networkx: approximate with distance-based estimate.

### Graph payload

The pickle file contains:
- `graph`: networkx DiGraph (or cuGraph on DGX)
- `resources`: DataFrame of resource nodes with `node_id`
- `transit`: DataFrame of transit nodes with `node_id`
- `tracts`: DataFrame of tract nodes with `node_id`
- `edges`: DataFrame of all edges with `src, dst, weight, edge_type`
- `backend`: "networkx" or "cugraph"
- `offsets`: node ID range boundaries

---

## SPO Triples & txt2kg

### SPO Triples (`data/triples.parquet`)

**328,868 subject-predicate-object triples** extracted from the resource mart, NYPD, 311, and PLUTO data. Built by `build_triples.py`.

Examples:
```
(shelter_42, HAS_TYPE, shelter)
(shelter_42, IN_BOROUGH, BK)
(shelter_42, HAS_SAFETY_SCORE, 0.85)
(shelter_42, NEAR_TRANSIT, transit_15)
(bbl_3001230045, IS_OVERFLOW_CANDIDATE, True)
```

81 unique predicates across 11 source datasets.

### txt2kg — 311 Complaint Extraction (`data/triples_311.parquet`)

**1.2 million triples** extracted from 311 complaint text using rule-based NLP.

The 311 dataset has `complaint_type` and `descriptor` columns (free text). We map these to structured triples:

```python
# Rule-based mapping (engine/txt2kg.py)
complaint_type="HEAT/HOT WATER" → issue_category="heating", severity="high"
complaint_type="PLUMBING"       → issue_category="plumbing", severity="medium"
complaint_type="PAINT/PLASTER"  → issue_category="structural", severity="low"
```

Generated triples:
```
(123_Main_St_BK, HAS_ISSUE, heating)
(123_Main_St_BK, ISSUE_SEVERITY, high)
(123_Main_St_BK, AFFECTS_POPULATION, residents)
(123_Main_St_BK, ISSUE_DATE, 2024-03-15)
```

**43,000 unique addresses** with categorized issues. This enriches the knowledge graph with real-world conditions at specific locations.

On DGX: swaps to NVIDIA txt2kg for full NLP entity/relation extraction from complaint text.

### KGE Embeddings (`engine/embeddings.py`)

40-dimensional feature vectors per resource for similarity search:
- One-hot encoded: resource type (19 dims) + borough (5 dims)
- Numeric: safety score, quality score, transit distance
- Crime/complaint counts (log-scaled)
- Binary: ADA accessible, has capacity data

Use case: "Find resources similar to this shelter" → cosine similarity over embedding vectors → returns 5 most similar resources. Shown in the UI as "Similar Resources" expander.

---

## Query Flow

### Step 1: Planning (`pipeline/planner.py`)

The LLM receives a system prompt with:
- 4 valid intents: `lookup`, `needs_assessment`, `simulate`, `explain`
- Social worker reasoning rules (priority ordering, eligibility screening)
- 9 few-shot examples covering all demo queries

**Example input:** "I'm Tina, 4 kids ages 12-16, income $28K, staying with my sister in Flatbush but she's kicking me out next week"

**Example output (JSON plan):**
```json
{
  "intent": "needs_assessment",
  "client_profile": {
    "household_size": 5,
    "children_ages": [12, 13, 15, 16],
    "annual_income": 28000,
    "borough": "BK",
    "current_situation": "doubled up, facing eviction"
  },
  "identified_needs": [
    {"category": "housing", "priority": 1, "reasoning": "Imminent loss of housing"},
    {"category": "benefits_screening", "priority": 2, "reasoning": "Income $28K for family of 5 = likely eligible for SNAP, Medicaid"},
    {"category": "school_continuity", "priority": 3, "reasoning": "4 school-age children need stable enrollment"}
  ],
  "resource_searches": [
    {"resource_types": ["shelter"], "filters": {"borough": "BK"}, "limit": 5},
    {"resource_types": ["benefits_center"], "filters": {"borough": "BK"}, "limit": 3},
    {"resource_types": ["school"], "filters": {"borough": "BK"}, "limit": 3},
    {"resource_types": ["legal_aid"], "filters": {"borough": "BK"}, "limit": 3}
  ]
}
```

### Step 2: Execution (`pipeline/executor.py`)

The executor runs each resource search against the real data:
- Filters by resource type and borough
- Excludes user-reported resources (feedback loop)
- Sorts by safety_score descending, then quality_score
- Returns top N results

For simulation intents, dispatches to specialized functions:
- `cold_emergency`: finds available shelters + PLUTO overflow sites + nearby food banks
- `resource_gap`: computes resources per 100K population by borough
- `capacity_change`: simulates adding beds and computes coverage impact
- `migrant_allocation`: matches people to shelter+food+school clusters with language preferences

### Step 3: Synthesis (`pipeline/synth.py`)

LLM takes the raw executor results and generates a natural language response:
- Formats resource lists with names, addresses, boroughs
- Adds eligibility information for benefits screening
- Prioritizes by the plan's priority ordering
- Generates a targeted follow-up question to unlock more resources

### Step 4: Verification (`pipeline/verify.py`)

Two-phase fact-checking:

**Phase 1 — Claim extraction:** LLM reads the synthesized answer and extracts factual claims (e.g., "Jewish Community Council shelter is in Brooklyn")

**Phase 2 — Verification:** Each claim is checked against the raw executor data:
- Name matches → VERIFIED
- Address matches → VERIFIED
- Borough matches → VERIFIED
- No match found → UNVERIFIED

Results shown as per-claim badges in the UI.

**Reasoning paths:** Multi-hop data provenance chains with confidence scores:
```
Data source (resource_mart.parquet) → confidence: 1.0
  ↓
Filter (resource_type=shelter, borough=BK) → confidence: 0.95
  ↓
Safety ranking (score=0.85, rank 2/350) → confidence: 0.85
  ↓
Cumulative confidence: 1.0 × 0.95 × 0.85 = 0.81
```

---

## LLM Integration

### Provider fallback ladder (`llm/client.py`)

The system tries LLM providers in order until one responds:

1. **NIM container** (localhost:8000) — NVIDIA's containerized inference, used on DGX with Nemotron-3-Nano-30B
2. **Claude Haiku** — Anthropic API, recommended for local development (fast, cheap, reliable JSON)
3. **GPT-4o-mini** — OpenAI API, fallback
4. **vLLM** (localhost:8001) — open-source inference server
5. **llama.cpp** (localhost:8080) — CPU inference for GGUF models
6. **OpenRouter** — aggregator with free Nemotron tier

### Anthropic SDK JSON trick

Claude sometimes wraps JSON in markdown code fences (` ```json ``` `), which breaks parsing. Fix: use **assistant-turn prefill** — start the assistant's response with `{` so Claude continues with raw JSON:

```python
# In client.py
if is_json_call:
    user_msgs = list(user_msgs) + [{"role": "assistant", "content": "{"}]
resp = ac.messages.create(model=provider.model, messages=user_msgs, ...)
result = "{" + resp.content[0].text  # re-attach the prefilled "{"
```

### API key requirement

At least one API key is required:
- `ANTHROPIC_API_KEY` — Claude Haiku ($0.25/M input, $1.25/M output)
- `OPENAI_API_KEY` — GPT-4o-mini
- `OPENROUTER_API_KEY` — free tier available

Or run a local model server (NIM, vLLM, llama.cpp) on localhost.

---

## Verification & Confidence

### Per-claim verification

Every answer gets fact-checked. The verifier LLM extracts claims and checks each against raw data:

```
✅ VERIFIED: "BronxWorks is located in the Bronx" — matches resource_mart row
✅ VERIFIED: "Safety score: 0.92" — matches safety_score column
⚠️ UNVERIFIED: "Open 24/7" — hours data not in database
```

### Confidence scoring

Each reasoning hop has a confidence score (0.6–1.0):
- Data source confidence: 1.0 (direct from parquet)
- Filter confidence: 0.9–0.95 (depends on filter specificity)
- Ranking confidence: 0.7–0.9 (depends on score distribution)
- Cumulative: multiply across hops

Color coding: 🟢 ≥0.7 | 🟡 0.5–0.7 | 🔴 <0.5

### Explain intent

Users can ask "why is the Bronx underserved?" → the system traverses the knowledge graph, computing multi-hop reasoning paths with per-step confidence, and presents both technical hop-by-hop view and plain-English summary.

---

## User Feedback Loop

The system supports **ground truth correction** — users report real-world issues and the system adapts.

### Flow

1. User sees recommendation: "Go to Shelter X at 123 Main St"
2. User goes there, finds it full
3. User types: "Shelter X is full"
4. System parses feedback (LLM identifies which resource + issue type)
5. Resource added to session exclusion list
6. Pipeline re-runs, excluding that resource
7. User gets alternative recommendations

### Issue types

| Issue | Trigger words | Effect |
|-------|--------------|--------|
| `full` | "full", "no beds", "no room", "capacity" | Exclude from results |
| `closed` | "closed", "shut", "not open" | Exclude from results |
| `wrong_address` | "wrong", "not there", "doesn't exist", "moved" | Exclude from results |
| `unsafe` | "unsafe", "dangerous", "scary" | Exclude from results |
| `other` | anything else | Exclude from results |

### Implementation

- `pipeline/feedback.py`: LLM parses free-form feedback to structured `{resource_name, issue, detail}`
- `pipeline/executor.py`: `set_excluded_resources()` filters out reported resources before any query
- `app.py`: session state tracks excluded resources + feedback log, cleared on "Start Over"

---

## NVIDIA Stack (DGX Spark)

On the DGX Spark (GB10, 128GB unified memory), CPU components swap to GPU-accelerated equivalents:

| Component | Local (CPU) | DGX (GPU) | Purpose |
|-----------|------------|-----------|---------|
| DataFrames | pandas | **cuDF** | Filter 7K resources in <1ms |
| Graph | networkx | **cuGraph** | SSSP, BFS, PageRank, Louvain |
| KNN | scipy KDTree | **cuML** | Spatial nearest-neighbor joins |
| Heatmaps | numpy | **cuPy** | Accessibility heatmaps |
| Optimization | greedy | **cuOpt** | VRP for resource allocation |
| LLM inference | Claude API | **NIM + Nemotron** | Local, private, sub-second |
| txt2kg | rule-based | **NVIDIA txt2kg** | Full NLP entity extraction |
| Compilation | — | **TensorRT-LLM** | Optimized inference engine |

### Memory footprint

| Layer | Size |
|-------|------|
| Resource mart (7.7K rows) | ~1.4 MB |
| PLUTO layer (857K lots) | ~38 MB |
| Knowledge graph | ~12 MB |
| SPO triples | ~7 MB |
| NYPD + 311 (raw, for scoring) | ~10 MB |
| **Local total** | **~70 MB** |
| + Nemotron-3-Nano-30B (Q4 GGUF) | ~16 GB |
| + Full NYPD/311/ACS on GPU | ~8 GB |
| + cuGraph full city graph | ~3 GB |
| + Simulation buffers | ~2 GB |
| **DGX total** | **~31 GB** |

The 128GB unified memory on DGX Spark means the full model + full city data + graph all live in memory simultaneously — no disk I/O during queries.

---

## File Structure

```
nyc_hack/
├── app.py                     # Streamlit UI — conversation-first layout
├── llm/
│   └── client.py              # LLM fallback ladder (NIM → Claude → GPT → OpenRouter)
├── pipeline/
│   ├── planner.py             # NL → JSON plan (4 intents, 9 few-shot examples)
│   ├── executor.py            # Execute plan against resource mart + graph
│   ├── synth.py               # Raw results → natural language answer
│   ├── verify.py              # Per-claim fact verification + reasoning paths
│   ├── feedback.py            # User-in-the-loop ground truth correction
│   └── clarify.py             # Multi-turn follow-up questions
├── engine/
│   ├── confidence.py          # Multi-hop graph traversal with confidence scoring
│   ├── embeddings.py          # KGE embeddings for similarity search
│   └── txt2kg.py              # 311 complaint → structured triples
├── data/                      # Generated data (included in repo)
│   ├── resource_mart.parquet  # 7,759 resources × 19 types
│   ├── pluto_layer.parquet    # 857K tax lots with overflow candidates
│   ├── graph.pkl              # Knowledge graph (networkx)
│   ├── triples.parquet        # 328K SPO triples
│   └── triples_311.parquet    # 1.2M triples from 311 complaints
├── stage/                     # Cleaned intermediate data (included in repo)
│   ├── shelters.parquet
│   ├── food_banks.parquet
│   ├── hospitals.parquet
│   ├── dohmh_facilities.parquet
│   ├── schools.parquet
│   ├── domestic_violence.parquet
│   ├── transit_stations.parquet
│   ├── nypd_complaints.parquet
│   ├── 311_complaints.parquet
│   ├── pluto.parquet
│   ├── benefits_centers.parquet
│   ├── dropin_centers.parquet
│   ├── nycha.parquet
│   └── cooling_centers.parquet
├── pull_all.py                # Download raw data from NYC Open Data
├── clean_all.py               # Clean raw → stage
├── build_mart.py              # Stage → resource mart
├── build_graph.py             # Mart → knowledge graph
├── build_triples.py           # Mart → SPO triples
├── ARCHITECTURE.md            # System architecture overview
├── DEMO_SCRIPT.md             # Demo presentation script
├── DEVPOST.md                 # Hackathon submission writeup
└── requirements.txt           # Python dependencies
```
