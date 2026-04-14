# NYC Social Services — Live Ops

An interactive 3D visualization of New York City's social-services network — every shelter, food bank, cooling center, community center, clinic, and school rendered as a glowing beacon, with live scenario animations that route synthetic demand to real resources and honestly report where the system runs out of capacity.

Built on **NYC Open Data** + `deck.gl` + `maplibre-gl`. Runs locally on a GTX 1060-class GPU (or any Mac with Metal support). No cloud, no API keys required for the core demo.

---

## What you're looking at

When the map loads, you see NYC at night. Hexagonal neon columns mark every social-services resource the city operates — color encodes type, height encodes capacity.

Click one of the bottom pills to fire a scenario:

| Pill | What it simulates | Real-world analogue |
|---|---|---|
| **Cold Snap** | 2,500 people need emergency warming-center placement across all five boroughs | NYC Code Blue declaration |
| **Migrant Bus** | 500 new arrivals dispersed to intake sites citywide | DHS Port Authority reception + borough buses |
| **Citywide Storm** | 4,000 concurrent routings — stress test | Mass event / simultaneous demand |
| **Reset** | Clear everything | — |
| **AUTO** | Master toggle: cycle the three scenarios continuously + orbit the camera | — |

Each scenario runs an honest greedy nearest-with-capacity allocator (scikit-learn `BallTree`) in Python — no LLM in the critical path — and returns:

- **Arcs + flowing neon trails**: every successfully routed person
- **Pulsing red dots**: "unmet" — people the shelter system had no capacity within a reasonable radius to serve. This is NYC's real story on bad nights.
- **Site columns that grow with load** — bright thick columns where demand is landing
- **Dimmed columns** — sites outside the active scenario's area (spotlighting what's in play)

A live stats HUD reports served / unmet / active sites / average distance / latency as the animation plays.

### Type filter

The dropdown at the top lets you isolate a resource type (e.g. show only shelters, or only food banks). When you fire a scenario, the filter auto-syncs to whatever primary resource type that scenario targets — so the dropdown and the map always stay coherent.

### Other pages still here

This repo started as a hackathon project and the original flows are intact:

- **User portal** (`/`) — natural-language query → resource + eligibility + case registration. Uses local llama3 via Ollama when available.
- **Admin portal** (`:9001`) — caseload dashboard, per-case AI briefings, form-fill-from-ID (llama3.2-vision → pre-filled NYC LDSS-4826 SNAP and DOH-4220 Medicaid PDFs).
- **Live Ops Demo** — this README's main subject.

---

## Quickstart

### Linux (tested on Ubuntu 22.04)

```bash
# 1. Clone + cd
git clone https://github.com/Nishant-ZFYII/nyc_hack.git
cd nyc_hack
git checkout linkedin-demo

# 2. Tesseract (fallback OCR for the form-filler — not strictly needed for live-ops)
sudo apt-get install -y tesseract-ocr

# 3. Python env
conda create -n nyc_hack python=3.11 -y
conda activate nyc_hack
pip install -r requirements.txt

# 4. (Optional) Ollama for the LLM agent drawer
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /tmp/ollama.log 2>&1 &
ollama pull llama3

# 5. Seed demo cases (only needed if you'll poke the admin portal)
python seed_demo_cases.py

# 6. Launch both portals
uvicorn server:app       --host 127.0.0.1 --port 9000 --workers 1 &
uvicorn admin_server:app --host 127.0.0.1 --port 9001 --workers 1 &

# 7. Open it
xdg-open http://127.0.0.1:9000/   # then click "Live Ops Demo"
xdg-open http://127.0.0.1:9001/   # admin view
```

### macOS (tested requirements — should Just Work)

The codebase uses only cross-platform libraries — no Linux-specific syscalls, no CUDA requirement, no systemd.

```bash
# 1. Clone
git clone https://github.com/Nishant-ZFYII/nyc_hack.git
cd nyc_hack
git checkout linkedin-demo

# 2. Tesseract
brew install tesseract

# 3. Python env (via miniconda or directly)
conda create -n nyc_hack python=3.11 -y
conda activate nyc_hack
pip install -r requirements.txt

# 4. (Optional) Ollama
brew install ollama
ollama serve > /tmp/ollama.log 2>&1 &
ollama pull llama3

# 5. Seed demo cases
python seed_demo_cases.py

# 6. Launch
uvicorn server:app       --host 127.0.0.1 --port 9000 --workers 1 &
uvicorn admin_server:app --host 127.0.0.1 --port 9001 --workers 1 &

# 7. Open it
open http://127.0.0.1:9000/
open http://127.0.0.1:9001/
```

**Notes for Mac users:**

- On Apple Silicon (M1/M2/M3), Ollama uses Metal for GPU acceleration automatically — llama3:8b runs at ~30 tok/s on an M1 Pro.
- `pyarrow` and `scikit-learn` have native arm64 wheels — pip install is fast, no compilation.
- deck.gl + maplibre run entirely in your browser (Chrome/Safari) via WebGL2 — no host-GPU config needed.
- If `brew install tesseract` is missing a language pack: `brew install tesseract-lang`.

### Windows

Not personally tested. Should work via WSL2 (follow the Linux path) or Miniconda for Windows directly. The only Windows-specific adjustment would be `curl`'s `--output` style and using `start` instead of `xdg-open`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             BROWSER (client)                             │
│                                                                          │
│   ┌───────────────────┐       ┌────────────────────────────────────┐    │
│   │ scenario_player.js│       │ maplibre-gl  (dark-matter basemap) │    │
│   │                   │──────▶│                                    │    │
│   │ - state loop      │       │ ┌────────────────────────────────┐ │    │
│   │ - auto-orbit      │       │ │ deck.gl MapboxOverlay          │ │    │
│   │ - filter dropdown │       │ │                                │ │    │
│   │ - phase captions  │       │ │ • ColumnLayer (sp-city)        │ │    │
│   │ - animated HUD    │       │ │   every resource as neon beacon│ │    │
│   │ - tooltip         │       │ │ • ColumnLayer (sp-sites)       │ │    │
│   └─────────┬─────────┘       │ │   active scenario sites, fat   │ │    │
│             │                 │ │ • ScatterplotLayer (sp-unmet)  │ │    │
│             │   fetch         │ │   pulsing red dots = unplaced  │ │    │
│             │                 │ │ • TripsLayer (sp-trips)        │ │    │
│             ▼                 │ │   animated neon flow along arcs│ │    │
│   ┌───────────────────┐       │ └────────────────────────────────┘ │    │
│   │   fetch + JSON    │       └────────────────────────────────────┘    │
│   └─────────┬─────────┘                                                  │
└─────────────┼────────────────────────────────────────────────────────────┘
              │
              ▼  HTTP on 127.0.0.1:9000
┌──────────────────────────────────────────────────────────────────────────┐
│                         FastAPI (server.py, CPU-only)                    │
│                                                                          │
│   GET  /                          →   frontend/index.html (live ops)     │
│   GET  /frontend/*                →   static assets                      │
│   GET  /api/resources             →   all NYC geocoded resources         │
│   POST /api/scenario/{name}       →   run cold/migrant/storm/reset       │
│   GET  /api/scenario/state        →   last scenario snapshot             │
│   GET  /api/vulnerability         →   (subtle context layer, optional)   │
│   POST /api/query  /api/agent/*   →   legacy chat + NeMo agent (opt)     │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │                pipeline/scenarios.py                          │       │
│   │                                                              │       │
│   │  • _synth_demand_in_borough — real on-land anchors (schools, │       │
│   │    childcare, community centers), ±150m jitter, NO water     │       │
│   │  • _demand_split            — weight by 2020 census share    │       │
│   │  • _greedy_allocate         — sklearn BallTree k-NN, strict  │       │
│   │    15 km physical cap, returns (arcs, sites, unmet)          │       │
│   │  • cold_emergency | migrant_bus | citywide_storm | reset     │       │
│   └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │ data/*.parquet   │   real NYC Open Data
                      │                  │
                      │ resource_mart    │   7,759 resources, 19 types
                      │ pluto_layer      │   857 K tax lots (legacy)
                      │ triples          │   328 K + 1.2 M SPO (legacy)
                      └──────────────────┘
```

---

## What's real, what's synthetic

Full transparency. The source and quality of everything visible:

| Element | Source | Notes |
|---|---|---|
| NYC resource locations + types | [NYC Open Data](https://opendata.cityofnewyork.us/) — DOHMH facilities, HRA HOPWA, DOE schools, DHS shelters, NYPD precincts | Cleaned into `data/resource_mart.parquet`. Borough, address, lat/lon all real. |
| Resource **capacity** | Mix of real values + imputed defaults | Where the city publishes bed counts we use them. Where not, we substitute a per-type default (shelter=60, hospital=300, etc.). Imputed values clearly flagged in code. |
| Demand points ("people") | **Synthetic** | Sampled from on-land anchors (schools/childcare/community centers) + jittered ±150 m. These are *not* real people — they're a proxy for where the population is. |
| Demand distribution across boroughs | 2020 US Census population share | BK 31.1%, QN 27.2%, MN 19.6%, BX 16.9%, SI 5.7% |
| Routing | Greedy nearest-with-capacity (sklearn BallTree), 15 km cap | Real algorithm, real latencies |
| Unmet demand (red dots) | Computed honestly — reflects actual NYC shelter-desert reality in Staten Island when demand exceeds local capacity | |

What this project is **not**:
- Not a live feed of actual NYC shelter occupancy (DHS doesn't publish that publicly).
- Not a booking system — don't use it to dispatch a real person.
- Not trained on any personal data.

---

## Tech stack

**Frontend**
- `maplibre-gl@3.6` — dark-matter basemap
- `deck.gl@latest` (via CDN) — `ColumnLayer`, `ScatterplotLayer`, `TripsLayer`, `HeatmapLayer`
- Pure vanilla JS — no React, no build step

**Backend**
- `fastapi` + `uvicorn` — single-worker async
- `pandas` + `pyarrow` — parquet IO
- `scikit-learn` — `BallTree` nearest-neighbor
- `scipy`, `networkx` — legacy pipeline modules

**Data + ML (optional, for the non-live-ops features)**
- `nemoguardrails` — PII/jailbreak/crisis filter on agent queries
- `nvidia-nat` + `nvidia-nat-langchain` — NeMo Agent Toolkit ReAct agent
- `pdfplumber` + `pypdf` + `reportlab` — PDF form filler
- `pytesseract` + `pillow` — fallback OCR when vision model unavailable
- `ollama` (external) — serves `llama3` + `llama3.2-vision` for agent + form-filler

Nothing here requires a CUDA GPU. NVIDIA RAPIDS libs are *not* installed by default; the project gracefully degrades to CPU equivalents.

---

## Key files

| Path | Role |
|---|---|
| `frontend/index.html` | User portal + Live Ops page |
| `frontend/admin.html` | Admin portal |
| `frontend/scenario_player.js` | **The live-ops engine** — map init, TripsLayer flow, HUD, captions, dropdown filter, bbox spotlight |
| `frontend/theme.css` | Dark-neon visual theme tokens |
| `frontend/admin_map.js` | Admin caseload 3D map |
| `frontend/agent_drawer.js` | NeMo Agent right-drawer |
| `server.py` | FastAPI user portal |
| `admin_server.py` | FastAPI admin portal |
| `pipeline/scenarios.py` | **The allocator** — demand synthesis, population-weighted split, BallTree greedy, unmet handling |
| `pipeline/form_filler.py` | llama3.2-vision OCR + PDF overlay |
| `pipeline/ops_snapshot.py` | Admin live-ops aggregator |
| `pipeline/cases.py` | Case persistence (JSON files in `data/cases/`) |
| `agent/register.py` | NeMo Agent Toolkit tool registration |
| `llm/client.py` | LLM provider fallback ladder |
| `seed_demo_cases.py` | Seeds 8 curated demo cases for the admin portal |
| `tests/test_scenarios.py` | Pytest sanity checks on the allocator |
| `data/resource_mart.parquet` | 7,759 NYC resources with coords |
| `data/cases/*.json` | Persisted case files |

---

## Branches

- `main` — stable, pre-live-ops version
- `linkedin-demo` — **this branch**, current work, the 3D live-ops viz described above

---

## Known limitations + intentional trade-offs

- **Capacity numbers** for shelters are partially imputed — NYC doesn't publish real-time bed counts. The relative differences between boroughs are real; absolute numbers are approximations.
- **Demand is synthesized.** We don't have anonymized real case-level data (and ethically shouldn't). The spatial distribution of demand follows population share + real on-land anchors.
- **No real-time shelter feed.** DHS Daily Report is aggregate, not per-facility.
- **Ollama models are ~5 GB each.** If you skip the agent drawer / form filler, you don't need them.
- **Performance**: 28 K resource beacons + 1,500 sites + up to 4 K concurrent trips renders at 55–60 fps on a GTX 1060 / M1 Pro. On older integrated GPUs it may drop to 25–30 fps.

---

## Testing

```bash
# Pytest (sanity checks on the allocator — 8 tests, all green)
PYTHONPATH= python -m pytest tests/test_scenarios.py -q

# Manual smoke test
curl -s http://127.0.0.1:9000/api/scenario/cold_emergency | jq '.stats'
# expected: {"served": 2511, "unmet": 0, "avg_km": 0.4, "elapsed_ms": ~1800}
```

---

## Credits + data sources

- NYC Open Data — [opendata.cityofnewyork.us](https://opendata.cityofnewyork.us/)
- PLUTO (Primary Land Use Tax Lot Output) — NYC Department of City Planning
- Coalition for the Homeless — [coalitionforthehomeless.org](https://www.coalitionforthehomeless.org/) — context + real statistics
- NYC DHS — Daily Shelter Census aggregate reports
- [StreetLives NYC / YourPeer](https://www.streetlives.nyc/) — peer-designed reference for how this space works in practice
- [City Limits](https://citylimits.org/) — investigative reporting on the NYC shelter intake process
- [deck.gl showcase](https://deck.gl/showcase) + [Peter Beshai's point-animation blog](https://peterbeshai.com/blog/2019-08-10-deckgl-point-animation/) — animation technique references
- [CartoDB dark-matter](https://carto.com/basemaps/) basemap style

---

## License

MIT — see `LICENSE`.

The underlying NYC Open Data is subject to NYC's own terms of use.

---

## Author

Built by [@Nishant-ZFYII](https://github.com/Nishant-ZFYII). This started as a submission to NVIDIA Spark Hack NYC 2026 and was rebuilt as a portfolio piece for LinkedIn.

If you use any part of this for your own civic-tech project, no attribution required but I'd love to hear about it.
