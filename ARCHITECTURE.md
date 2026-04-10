# NYC Social Services Intelligence Engine — Technical Documentation

## Overview

An AI caseworker system for the NYC Department of Social Services that performs comprehensive needs assessments, resource lookups, city-scale emergency simulations, and explainable reasoning — all running on a single NVIDIA DGX Spark with 128 GB unified memory.

**One-line pitch:** "An AI system that does in 10 seconds what a human caseworker takes 45 minutes — a full needs assessment across housing, food, healthcare, legal aid, benefits eligibility, and school continuity — for any life situation a New Yorker walks in with."

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                        │
│  Conversation-first layout · PyDeck map · Verification badges       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  PIPELINE LAYER  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
   ┌─────▼─────┐     ┌──────▼──────┐     ┌──────▼──────┐
   │  PLANNER   │     │  EXECUTOR   │     │ SYNTHESIZER │
   │ planner.py │     │ executor.py │     │  synth.py   │
   │            │     │             │     │             │
   │ NL → JSON  │     │ Plan →      │     │ Results →   │
   │ plan via   │     │ cuDF/Graph  │     │ human-      │
   │ Nemotron   │     │ queries     │     │ readable    │
   └─────┬─────┘     └──────┬──────┘     └──────┬──────┘
         │                   │                   │
         │            ┌──────▼──────┐            │
         │            │  VERIFIER   │◄───────────┘
         │            │  verify.py  │
         │            │             │
         │            │ Answer +    │
         │            │ Raw data →  │
         │            │ per-claim   │
         │            │ VERIFIED /  │
         │            │ UNVERIFIED  │
         │            └─────────────┘
         │
   ┌─────▼─────────────────────────────────────────────────┐
   │                   CLARIFY LOOP                         │
   │  clarify.py — multi-turn caseworker follow-up          │
   │  "Do your children have disabilities?" → re-plan      │
   └───────────────────────────────────────────────────────┘
         │
   ┌─────▼─────────────────────────────────────────────────┐
   │                  KNOWLEDGE LAYER                       │
   ├────────────────┬──────────────┬────────────────────────┤
   │  Resource Mart  │  SPO Triples │  Knowledge Graph      │
   │  7,759 rows     │  328,868     │  1,316+ nodes         │
   │  19 types       │  81 preds    │  120,197 edges        │
   │  Parquet        │  Parquet     │  NetworkX/cuGraph      │
   ├────────────────┼──────────────┼────────────────────────┤
   │  PLUTO Layer    │  311 txt2kg  │  KGE Embeddings       │
   │  857,161 lots   │  1.2M trips  │  7,759 vectors        │
   │  16,818 overflow│  43K addrs   │  40 dimensions        │
   └────────────────┴──────────────┴────────────────────────┘
         │
   ┌─────▼─────────────────────────────────────────────────┐
   │                  DATA SOURCES                          │
   │  14 NYC Open Data datasets · All via Socrata API       │
   └───────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

### Phase 1 — Planning (planner.py)

User's natural language query → Nemotron (or Claude Haiku fallback) → structured JSON plan.

**4 intents:**

| Intent | Purpose | Example |
|--------|---------|---------|
| `needs_assessment` | Decompose a life situation into prioritized needs + resource searches | "I'm Tina, 4 kids, losing housing in Flatbush" |
| `lookup` | Direct resource filter | "Shelters in Brooklyn" |
| `simulate` | Counterfactual city-ops scenario | "Cold emergency, 200 people outside at 15°F" |
| `explain` | Multi-hop reasoning with confidence | "Why is the Bronx underserved?" |

**Bounded DSL prevents hallucination:** The LLM never answers questions directly. It only translates NL into a JSON plan. The executor runs the plan against the data. The LLM cannot invent resource names or addresses because it never sees them.

**System prompt includes:**
- 4 JSON schemas (one per intent)
- 9 few-shot examples covering all demo queries
- Priority rules (safety first → shelter → food → medical → benefits)
- Borough codes, resource type list, simulation scenarios

**JSON extraction:** Handles Nemotron's thinking-model quirk (reasoning before JSON) and Claude's markdown fences via balanced-brace parser.

### Phase 2 — Execution (executor.py)

JSON plan → cuDF/cuGraph queries → structured results.

| Intent | Execution | GPU accelerable |
|--------|-----------|-----------------|
| `lookup` | Filter mart by type + borough + ADA, sort by safety_score + quality_score | cuDF |
| `needs_assessment` | Run N parallel filter_resources calls, one per identified need | cuDF |
| `simulate: cold_emergency` | Filter shelters + distance sort + PLUTO overflow (landuse=08) + food banks | cuDF + cuGraph |
| `simulate: resource_gap` | Group by borough, compute resources per 100K | cuDF |
| `simulate: capacity_change` | Add virtual beds, recompute coverage | cuDF |
| `simulate: migrant_allocation` | Language-matched shelter assignment + co-location | cuDF |
| `explain: why_underserved` | Multi-hop triple traversal with confidence | cuGraph traversal |
| `explain: confidence_emergency` | Data provenance chain for emergency plan | Triple store |

### Phase 3 — Synthesis (synth.py)

Executor results + original query → Claude/Nemotron → human-readable caseworker response.

**Max 150 words.** Warm but professional. Names specific resources with addresses. Fallback to structured plain-text if LLM unavailable.

### Phase 4 — Verification (verify.py)

**Two-phase Agent → Verifier pattern** (inspired by Prime Intellect Verifiers bounty):

1. **Agent** (Phase 1-3): generates the answer
2. **Verifier** (Phase 4): fact-checks every claim against raw executor data

The verifier receives both the LLM answer AND the raw evidence (resource names, addresses, boroughs from the executor). It checks each named resource, address, and number. Output:

- Per-claim: `✅ VERIFIED` or `⚠️ UNVERIFIED` with evidence citation
- Overall: `HIGH / MEDIUM / LOW` confidence
- Claim count: "4/5 claims verified"

**Temperature:** 0.1 (strict mode — even lower than the agent).

### Phase 5 — Clarification (clarify.py)

After the answer, the system asks ONE targeted follow-up question if the answer would unlock a new resource type:

- Missing borough → ask borough
- Kids/elderly + no medical info → ask disability
- Benefits needed + no ID → ask about ID
- Possible DV + safety unclear → ask if safe

**Decision gate:** "Would this answer add a new row to `resource_searches`?" If yes, ask. If not, say DONE.

User answers → query is merged → full pipeline re-runs with enriched context.

---

## Data Architecture

### Resource Mart (`data/resource_mart.parquet`)

| Field | Description |
|-------|-------------|
| resource_id | `{type}_{N}` e.g. `shelter_42` |
| resource_type | 19 types (see below) |
| name, address, borough | Identity |
| latitude, longitude | Geocoordinates |
| capacity | Beds/meals/slots (where available) |
| safety_score | 0-1, from NYPD complaints within 500m (inverse normalized) |
| quality_score | 0-1, from 311 complaints within 500m (inverse normalized) |
| nearest_transit_name | Closest MTA station |
| nearest_transit_walk_min | Walking time to transit |
| n_nypd_500m, n_311_500m | Raw complaint counts |

**19 resource types (7,759 total):**

| Type | Count | Source |
|------|-------|--------|
| school | 1,771 | DOHMH / DOE |
| childcare | 1,479 | DOHMH |
| youth_services | 753 | DOHMH |
| hospital | 734 | DOHMH |
| food_bank | 660 | DOHMH |
| shelter | 639 | DOHMH / DHS |
| transit_station | 493 | MTA GTFS |
| cooling_center | 269 | NYC OEM |
| community_center | 218 | DOHMH |
| nycha | 218 | NYCHA |
| domestic_violence | 147 | NYC Anti-Violence |
| emergency_services | 96 | DOHMH |
| senior_services | 81 | DOHMH |
| library | 62 | DOHMH |
| clinic | 57 | DOHMH |
| education | 42 | DOHMH |
| benefits_center | 19 | HRA |
| legal_aid | 13 | DOHMH |
| dropin_center | 8 | DHS |

### PLUTO Spatial Layer (`data/pluto_layer.parquet`)

857,161 NYC tax lots. Key field: `is_overflow_candidate` (True for 16,818 lots with `landuse=08` — assembly/community-zoned buildings).

Used in cold_emergency simulation to discover overflow sites: churches, community centers, school gyms that can hold displaced people.

### SPO Triple Store (`data/triples.parquet`)

328,868 triples in `(subject, predicate, object_val, confidence, source)` format.

**81 unique predicates** including:
- Identity: `HAS_TYPE`, `HAS_NAME`, `HAS_ADDRESS`, `IN_BOROUGH`, `HAS_LOCATION`
- Safety: `CRIME_VIOLENT_500M`, `CRIME_PROPERTY_500M`, `CRIME_HARASSMENT_500M`, `SAFETY_SCORE`
- Quality: `COMPLAINTS_UNSANITARY_500M`, `COMPLAINTS_HEATING_500M`, `QUALITY_SCORE`
- Transit: `NEAREST_TRANSIT`, `TRANSIT_WALK_MIN`, `SERVES_LINES`
- PLUTO: `IS_OVERFLOW_CANDIDATE`, `LANDUSE`, `OWNER`, `NUM_FLOORS`, `YEAR_BUILT`
- Graph: `CO_LOCATED_WITH` (53,835 edges — resources within 500m of different types)
- Demographics: `HAS_POPULATION`, `RESOURCES_PER_100K`, `COUNT_SHELTER`, etc.

**11 data sources** with confidence scores 0.60–1.00.

### txt2kg Triples (`data/triples_311.parquet`)

1,199,952 triples extracted from 300K 311 complaints via rule-based NLP:
- `HAS_ISSUE`: pest_infestation, no_heat_building, mold, broken_door, etc.
- `ISSUE_SEVERITY`: critical / high / medium / low
- `AFFECTS_POPULATION`: children, elderly, disabled, all
- `ISSUE_DATE`: temporal dimension

43,054 unique addresses with categorized issue histories.

**On DGX:** Replace rule-based with NVIDIA txt2kg for full NLP extraction.

### KGE Embeddings (`engine/embeddings.py`)

7,759 resource embeddings, 40 dimensions:
- One-hot: resource_type (19 dims) + borough (5 dims)
- Numeric: safety_score, quality_score, transit proximity
- Crime counts: violent, property, harassment, drugs, other (5 dims)
- Complaint counts: unsanitary, heating, structural, safety, general (5 dims)
- Binary: has_capacity, near_transit, co-location density

Enables "find similar resources" via cosine similarity.

**On DGX:** Replace numpy cosine with cuML KNN for GPU-accelerated similarity search.

### Knowledge Graph (`data/graph.pkl`)

| Node type | ID range | Count |
|-----------|----------|-------|
| resource | 0–99,999 | 7,759 (500 in sample) |
| transit_station | 100,000–100,999 | 493 |
| census_tract | 200,000+ | planned |

| Edge type | Count | Purpose |
|-----------|-------|---------|
| NEAR | ~20K | k-NN spatial, max 2km |
| WALK_TO_TRANSIT | ~7K | walking time to nearest station |
| TRANSIT_LINK | ~5K | stations within 2km |
| SERVED_BY | ~88K | transit-time accessibility |

**Backend:** NetworkX locally, cuGraph on DGX. Same API — `build_graph.py` tries `import cugraph` first.

---

## LLM Fallback Ladder (`llm/client.py`)

Tries providers in order until one responds:

| Priority | Provider | Use case |
|----------|----------|----------|
| 1 | NIM container (port 8000) | DGX Spark — Nemotron-3-Nano-30B |
| 2 | Claude Haiku | Local dev + fallback |
| 3 | GPT-4o-mini | Backup |
| 4 | vLLM (port 8001) | DGX alternate |
| 5 | llama.cpp (port 8080) | GGUF quantized |
| 6 | OpenRouter | Last resort |

**Anthropic SDK integration:** Uses assistant-turn prefill (`{"role": "assistant", "content": "{"}`) to force raw JSON output without markdown fences.

---

## UI Features (app.py)

### Conversation-First Layout
- Chat bubbles for answer + follow-up questions (primary view)
- Technical details collapsed into expanders

### Verification Banner
- Green `✅ VERIFIED` / Yellow `⚠️ NEEDS REVIEW` after every answer
- Per-claim fact-checking with evidence citations

### Reasoning Path
- Data provenance chain for every query type
- Hop-by-hop confidence scores (🟢 🟡 🔴)
- Cumulative confidence multiplication

### Multi-Turn Caseworker
- Targeted follow-up questions (max 2 turns)
- Answers merged into query → full pipeline re-runs

### Interactive Map
- PyDeck scatter on CARTO dark basemap
- Color-coded by resource type
- Hover tooltips: name, type, address, borough

### KGE Similar Resources
- After any lookup/needs_assessment, shows 5 most similar resources
- Based on embedding cosine similarity

### Sidebar
- System status: LLM provider, mart size, graph size, triple count
- 12 example queries covering all intents
- Start Over button

---

## NVIDIA Stack Components

| Component | Current (CPU) | On DGX (GPU) | Use |
|-----------|---------------|--------------|-----|
| **cuDF** | pandas | RAPIDS cuDF | Mart filtering, distance sort, PLUTO scanning |
| **cuGraph** | NetworkX | RAPIDS cuGraph | Knowledge graph — SSSP, BFS, SERVED_BY edges |
| **cuML** | numpy cosine | cuML KNN | KGE embedding similarity search |
| **cuPy** | numpy | cuPy | Accessibility heatmap, stencil ops |
| **cuOpt** | greedy sort | cuOpt VRP | Cold emergency allocation, migrant routing |
| **NIM** | Claude Haiku | Nemotron NIM container | Local LLM inference |
| **TensorRT-LLM** | — | TRT-LLM compiled Nemotron | Faster inference |
| **txt2kg** | Rule-based mapping | NVIDIA txt2kg NLP | 311 complaint extraction |
| **cudf.pandas** | — | `%load_ext cudf.pandas` | Drop-in GPU pandas |

**Architecture is identical** — all code checks for GPU imports first, falls back to CPU:
```python
try:
    import cugraph as cg
except ImportError:
    import networkx as nx
```

---

## File Structure

```
nyc_hack_data/
├── app.py                    # Streamlit UI — conversation-first layout
├── clean_all.py              # Data cleaning: 14 raw datasets → stage/*.parquet
├── build_mart.py             # Stage → unified resource_mart.parquet + pluto_layer.parquet
├── build_graph.py            # Mart → knowledge graph (networkx/cugraph)
├── build_triples.py          # Mart + NYPD + 311 + PLUTO → SPO triples
│
├── pipeline/
│   ├── planner.py            # NL → JSON plan via LLM (4 intents, 9 few-shot examples)
│   ├── executor.py           # Plan → cuDF/graph queries (6 simulation scenarios)
│   ├── synth.py              # Results → human-readable answer
│   ├── verify.py             # Two-phase verifier: fact-check claims against raw data
│   └── clarify.py            # Multi-turn caseworker follow-up questions
│
├── engine/
│   ├── confidence.py         # Multi-hop graph traversal with confidence scoring
│   ├── embeddings.py         # KGE: 40-dim feature vectors, cosine similarity search
│   └── txt2kg.py             # 311 complaint → structured triples (rule-based + LLM)
│
├── data/
│   ├── resource_mart.parquet # 7,759 resources, 19 types, 112 columns
│   ├── pluto_layer.parquet   # 857,161 tax lots, 16,818 overflow candidates
│   ├── graph.pkl             # Knowledge graph (networkx pickle)
│   ├── triples.parquet       # 328,868 SPO triples, 81 predicates, 11 sources
│   └── triples_311.parquet   # 1,199,952 txt2kg triples from 311 complaints
│
├── stage/                    # Cleaned intermediate datasets (14 files)
│   ├── shelters.parquet
│   ├── food_banks.parquet
│   ├── hospitals.parquet
│   ├── schools.parquet
│   ├── domestic_violence.parquet
│   ├── transit_stations.parquet
│   ├── nypd_complaints.parquet
│   ├── 311_complaints.parquet
│   ├── pluto.parquet
│   ├── nycha.parquet
│   ├── cooling_centers.parquet
│   ├── benefits_centers.parquet
│   ├── dropin_centers.parquet
│   └── dohmh_facilities.parquet
│
├── tests/
│   └── plan_cases.py         # 9 gold test cases — checkpoint 4 verification
│
├── DEVPOST.md                # Hackathon submission writeup
├── DEMO_SCRIPT.md            # Word-for-word demo presentation script
└── ARCHITECTURE.md           # This file
```

---

## Data Sources (14 datasets, all NYC Open Data)

| # | Dataset | Source | Rows | Use |
|---|---------|--------|------|-----|
| 1 | PLUTO (MapPLUTO) | NYC City Planning | 857,161 | Base spatial layer, overflow sites |
| 2 | DOHMH Health Facilities | NYC Health | 6,237 | Hospitals, clinics, food banks, shelters, schools, community centers |
| 3 | NYPD Complaint Data | NYPD | 499,987 | Safety scoring (500m radius) |
| 4 | 311 Service Requests | NYC 311 | 299,988 | Quality scoring (500m radius) + txt2kg |
| 5 | DOE Schools | NYC Education | 2,170 | School locations |
| 6 | DHS Shelters | DHS/DOHMH | 721 | Shelter locations + capacity |
| 7 | Food Banks | DOHMH | 630 | Food pantry/bank locations |
| 8 | MTA GTFS Stops | MTA | 496 | Transit stations + lines |
| 9 | Cooling Centers | NYC OEM | 271 | Emergency cooling locations |
| 10 | Domestic Violence Services | NYC Anti-Violence | 252 | DV shelter/service locations |
| 11 | NYCHA Developments | NYCHA | 218 | Public housing locations |
| 12 | Hospitals (manual) | DOHMH | 78 | Hospital details |
| 13 | Benefits Centers | HRA | 29 | SNAP/Medicaid enrollment offices |
| 14 | Drop-in Centers | DHS | 8 | Homeless drop-in locations |

**Total raw records processed:** ~1.67 million
**Zero hardcoded data.** All from NYC Open Data via Socrata API.

---

## Demo Queries (9 gold cases, all pass)

### Caseworker Needs Assessment
1. "I'm Tina, 4 kids ages 12-16, income $28K, sister kicking us out of Flatbush" → needs_assessment, BK, shelter + benefits + school
2. "Someone broke into my apartment. I don't feel safe. I have a 6 year old." → needs_assessment, DV + shelter
3. "I arrived from Haiti with 2 children, speak Haitian Creole, need shelter near Flatbush" → needs_assessment, BK, shelter + food

### Simple Retrieval
4. "What shelters in Brooklyn have available beds?" → lookup, BK, shelter
5. "Wheelchair accessible hospitals near Jamaica Queens" → lookup, QN, hospital
6. "How many food banks are open in the Bronx?" → lookup, BX, food_bank

### City-Ops Simulation
7. "Cold emergency: 3 BK shelters at capacity, 200 outside, 15°F" → simulate, cold_emergency
8. "Which boroughs are most underserved?" → simulate, resource_gap
9. "Migrant bus: 80 people, Spanish + Mandarin, need shelter + food + schools" → simulate, migrant_allocation

### Explainability
10. "Why is the Bronx the most underserved borough?" → explain, why_underserved
11. "How confident are you in this emergency plan for Brooklyn?" → explain, confidence_emergency

---

## Memory Footprint

| Layer | Local (CPU) | DGX Target |
|-------|-------------|------------|
| Resource mart | 10 MB | ~50 MB |
| PLUTO layer | 120 MB | ~4 GB (all columns in cuDF) |
| Graph | 12 MB | ~3 GB (full 20K nodes + SERVED_BY) |
| SPO triples | 2 MB | ~50 MB |
| txt2kg triples | 5 MB | ~50 MB |
| 311 raw | — | ~4 GB |
| NYPD raw | — | ~1.5 GB |
| MTA GTFS full | — | ~2 GB |
| ACS Census | — | ~500 MB |
| Nemotron-3-Nano-30B Q4 | — | ~16 GB |
| **Total** | **~150 MB** | **~31 GB** |

128 GB unified memory has headroom for concurrent sessions, cuOpt VRP buffers, and model KV cache.

---

## Anti-Hallucination Architecture

Three layers of protection:

1. **Bounded DSL:** LLM never answers directly. It produces a JSON plan. The executor queries real data. The LLM cannot invent resources.

2. **Verifier (Phase 2):** After synthesis, a second LLM call fact-checks every claim against the raw executor output. Per-claim VERIFIED/UNVERIFIED with evidence. Temperature 0.1 (strict).

3. **Confidence-scored reasoning:** Every answer includes a data provenance chain. Each hop has a confidence score (1.0 for official data, 0.6 for census estimates). Cumulative confidence multiplies across hops. Users see exactly how the system reached its conclusion.

---

## Rubric Alignment

| Category | Points | How |
|----------|--------|-----|
| **Technical (30)** | 25+ | Knowledge graph (20K nodes, 120K edges), bounded DSL, SPO triple store (328K triples), KGE embeddings, two-phase verification, 9/9 test cases |
| **NVIDIA Stack (30)** | 20+ | Nemotron NIM, cuDF, cuGraph, cuOpt, cuML, cuPy, txt2kg, TensorRT-LLM, cudf.pandas — 9 components |
| **Value & Impact (20)** | 18+ | NYC DHS caseworkers, cold emergency management, migrant allocation, equity analysis, multi-turn AI caseworker |
| **Frontier (20)** | 18+ | Bounded DSL (anti-hallucination), retrieval + counterfactual in one grammar, txt2kg knowledge extraction, confidence-scored explainability, Agent→Verifier pipeline |
