# NYC Social Services Intelligence Engine

## Inspiration

NYC has 33,000 DHS caseworkers. Each one spends 45 minutes per client doing what a good caseworker does: decomposing a life situation into a comprehensive needs assessment — housing, food, healthcare, benefits eligibility, school continuity, legal rights. Most clients don't know what to ask for. Most caseworkers don't have time to screen for everything.

We asked: what if every caseworker had instant access to every resource in the city, with real-time availability, safety routing, and eligibility screening — and could do a full needs assessment in 10 seconds?

That's not a chatbot. That's a city-scale operations engine. And it needs a DGX Spark to run it.

## What it does

The NYC Social Services Intelligence Engine is an AI caseworker for the NYC Department of Social Services. It handles three classes of queries:

**1. Needs Assessment (the core)**
A person describes their situation in plain English. The system does what a human caseworker does: decomposes the situation into prioritized needs, screens for benefits eligibility (SNAP <130% FPL, Medicaid <138% FPL), identifies resources the person didn't know to ask for, and returns a ranked, actionable plan with specific addresses.

*"I'm Tina, 4 kids ages 12-16, income $28K, my sister is kicking us out of her Flatbush apartment next week."*
→ Emergency shelter options sorted by distance from Flatbush + benefits enrollment centers + school continuity resources, synthesized into a caseworker-style response in under 5 seconds.

**2. City-Ops Simulation (the hero demo)**
Emergency coordinators run counterfactual scenarios against the live resource graph:
- *Cold emergency:* 3 shelters hit capacity, 200 people outside at 15°F → activate PLUTO assembly-zoned overflow sites (churches, community centers), sorted by distance to displaced population, with food distribution points
- *Resource gap:* Which boroughs are most underserved per 100K residents?
- *Capacity change:* What happens to coverage if we add 500 beds in the Bronx?
- *Migrant allocation:* 80 people arrive speaking Spanish and Mandarin → language-matched shelter assignment with food and school co-location

**3. Direct Lookup**
Caseworkers query the resource graph directly: wheelchair-accessible hospitals in Queens, food banks open in the Bronx, domestic violence shelters in Manhattan.

**Multi-turn AI caseworker**
After every answer, the system asks one targeted follow-up question to refine the recommendations — missing borough, disability status, language needs, ID availability. The conversation thread is the primary UI; technical details are collapsed.

## How we built it

### Architecture
```
NL query → Nemotron planner → cuDF/cuGraph/cuOpt executor → Nemotron synthesis → Verification
```

**Bounded DSL prevents hallucination.** Nemotron doesn't answer questions directly — it translates natural language into a structured JSON plan (`needs_assessment`, `lookup`, or `simulate`). The executor runs the plan against the knowledge graph. The synthesizer turns results into prose. The LLM never touches the data directly.

### The Knowledge Graph
**7,759 resources** across 19 types: shelters, food banks, hospitals, clinics, schools, benefits centers, domestic violence services, senior services, childcare, community centers, NYCHA housing, and more.

**857,161 PLUTO tax lots** as the base spatial layer — the source of overflow sites during cold emergencies (landuse=08: assembly/community-zoned buildings).

**Graph nodes:** resources (0–99,999) + transit stations (100,000+) + census tracts (200,000+)
**Graph edges:** NEAR (k-NN spatial), WALK_TO_TRANSIT, TRANSIT_LINK (MTA GTFS), IN_TRACT, SERVED_BY (transit-time weighted accessibility)

### NVIDIA Stack (11 components)
| Component | Use |
|-----------|-----|
| **Nemotron via Ollama (GPU)** | Local LLM inference on DGX Spark GB10 — query planning, synthesis, verification. 100% GPU, no cloud. |
| **RAPIDS cuDF** | GPU-accelerated DataFrame filtering on 7,759 resources. Replaces pandas — sub-millisecond queries. |
| **RAPIDS cuGraph** | 3.6M-edge knowledge graph built in 90 seconds. SSSP for transit accessibility, BFS for resource clusters. |
| **cuOpt** | VRP solver for cold emergency allocation (200 people across 8 shelters in 2 seconds) and migrant allocation (language + proximity constraints). |
| **cuPy** | GPU-accelerated spatial scoring — count NYPD/311 complaints within 500m of each resource using broadcasting. |
| **RAPIDS cuML** | KNN spatial joins (resource→resource NEAR edges), HDBSCAN for neighborhood clustering. |
| **TensorRT-LLM** | Installed for compiled Nemotron inference optimization. |
| **PyKEEN + PyTorch** | Real TransE knowledge graph embeddings trained on 328K triples (83K entities, 64 dimensions). Not hand-crafted features — learned representations. |
| **txt2kg** | Rule-based extraction of 1.2M structured triples from 311 complaint text (with NVIDIA txt2kg swap point for full NLP on DGX). |
| **deck.gl** | GPU-accelerated 3D map visualization with 7,759 resource markers, animated overlays, fly-to interactions. |
| **CUDA 13.0** | DGX Spark GB10 compute capability 12.1, 128 GB unified memory. |

### Why DGX Spark (128 GB unified memory)
The full in-memory state:
- 870K PLUTO lots in cuDF: ~4 GB
- Full MTA transit graph (GTFS): ~2 GB  
- 311 complaint history (quality signals): ~4 GB
- NYPD crime data (safety routing): ~1.5 GB
- ACS Census demographics per tract: ~500 MB
- Nemotron-3-Nano-30B weights: ~16 GB
- cuGraph knowledge graph + SERVED_BY edges: ~3 GB
- **Total: ~31 GB active**

This cannot run on a laptop. It barely fits on a Jetson (64 GB shared with OS). The DGX Spark's 128 GB unified memory is exactly why this works: the LLM, the graph, and all the city data live in one address space. No swapping, no cloud, no privacy risk. A city that handles 8 million residents' most sensitive data cannot send it to an API.

### Data Sources (all NYC Open Data)
- PLUTO (MapPLUTO) — 870K tax lots with zoning
- DOHMH Health Facilities — hospitals, clinics, community health centers
- DHS/DOHMH Human Services — shelters, food banks, drop-in centers
- MTA GTFS — 496 transit stations
- NYPD Complaint Data — safety scoring within 500m
- 311 Service Requests — quality scoring within 500m
- DOE Schools — 2,170 schools
- HRA Benefits Centers — SNAP/Medicaid/cash assistance enrollment sites
- NYC Domestic Violence Resources

## Challenges

**Data cleaning was the real bottleneck.** DHS shelter census data is aggregate (system-wide totals, not per-shelter). Food bank data was buried in a healthcare provider directory. NYCHA coordinates were encoded as MultiPolygon geometry, not lat/lon. Cooling center coordinates were in a different column format than expected. We resolved all of these before the hackathon clock started.

**Nemotron's thinking-model quirk.** Nemotron-3-Nano outputs reasoning text before JSON. We handle this with regex extraction of the last balanced JSON block, plus Anthropic assistant-turn prefill (`{`) to force raw JSON output from Claude during local dev.

**Hallucination prevention.** Early versions had the LLM answer questions directly — it would invent resource names and addresses. The bounded DSL architecture eliminates this: the LLM only produces a plan, never touches the data.

## Accomplishments

- Full pipeline running on DGX Spark GB10 with 11 NVIDIA components
- cuOpt VRP allocates 200 displaced people across shelters in 2 seconds
- cuGraph knowledge graph: 3.6M edges built in 90 seconds (30x more than local)
- PyKEEN TransE embeddings: 83,618 entities, 64 dimensions, trained on 328K triples
- 4 simulation scenarios (cold_emergency, resource_gap, capacity_change, migrant_allocation)
- Multi-turn AI caseworker with targeted follow-up questions
- User-in-the-loop feedback: report "shelter is full" → system excludes and finds alternatives
- Per-claim verification with confidence-scored reasoning paths
- 857K PLUTO lots as overflow site discovery — novel application of NYC tax data
- Custom dark dashboard with deck.gl 3D map + chat interface + guided search
- Zero hallucination: bounded DSL architecture, LLM never touches data directly

## What we learned

The hardest part wasn't the GPU code or the LLM. It was understanding what a real caseworker actually does — the priority ordering (safety first, then shelter, then food, then medical, then benefits, then legal), the eligibility rules, the things clients don't know to ask for. Getting that social work domain knowledge into the system prompt was the real work.

## What's next

- **Real-time DHS census:** DHS publishes daily shelter census data. Wiring this in gives live capacity numbers.
- **MTA real-time delays:** GTFS-RT integration for live transit routing.
- **NOAA weather triggers:** Automatic cold/heat emergency detection from forecast data.
- **Whisper voice input:** Caseworkers speak into the DGX, Whisper STT transcribes, pipeline processes.
- **Caseworker mobile interface:** The DGX runs at a command center; field workers need a thin client.
- **Dual-model verification:** Separate verifier LLM (Qwen-14B) cross-checks Nemotron's synthesis for disagreement detection.

## Built with

Python · FastAPI · NVIDIA Nemotron (Ollama, GPU) · RAPIDS cuDF · RAPIDS cuGraph · cuOpt · cuPy · RAPIDS cuML · TensorRT-LLM · PyKEEN (TransE KGE) · deck.gl · MapLibre · NetworkX (CPU fallback) · NYC Open Data (Socrata) · PLUTO · MTA GTFS · pandas · NumPy · SciPy · PyTorch · Anthropic Claude Haiku (local dev fallback)
