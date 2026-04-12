# NYC Social Services Intelligence Engine

An AI-powered caseworker + kiosk system for NYC Department of Social Services. Built for **NVIDIA Spark Hack NYC 2026**.

The system performs comprehensive needs assessments across housing, food, healthcare, legal aid, benefits eligibility, school continuity, and safety — using a **bounded DSL architecture** that prevents LLM hallucination. It tracks people across multiple visits (like a real caseworker), finds resources sorted by distance from where they are, and guides them step-by-step through their journey.

---

## 🏁 Quickstart for evaluators

Pick ONE path. The Docker path is simpler; the bare-metal path is faster on a machine you already trust.

### Path A — Docker (recommended)

See **[DOCKER.md](./DOCKER.md)** for the full walkthrough. Three commands:

```bash
git clone https://github.com/Nishant-ZFYII/nyc_hack.git && cd nyc_hack
docker compose up -d
docker compose logs -f ollama-init       # wait for "Models ready."
```

Then open [http://localhost:9000](http://localhost:9000) (user) and [http://localhost:9001](http://localhost:9001) (admin).

### Path B — Bare metal (no Docker)

Works on Linux/macOS. Tested on Ubuntu 22.04 and the NVIDIA DGX Spark.

#### 1. System prerequisites

```bash
# Python 3.10 or newer (3.12 recommended)
python3 --version

# Tesseract — fallback OCR (primary OCR uses llama3.2-vision via Ollama)
#   Ubuntu/Debian:
sudo apt-get install -y tesseract-ocr
#   macOS:
brew install tesseract
```

#### 2. Install Ollama + pull the two models

```bash
# Install Ollama: https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama with GPU enabled (required on DGX — defaults to CPU otherwise)
sudo systemctl stop ollama 2>/dev/null
OLLAMA_NUM_GPU=999 OLLAMA_KEEP_ALIVE=2h ollama serve > /tmp/ollama.log 2>&1 &

# Pull the two models the system uses (~25 GB, 10–20 min depending on bandwidth)
ollama pull llama3              # drives the ReAct agent
ollama pull llama3.2-vision:11b # reads ID photos for the form-fill feature

# Verify
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

#### 3. Clone + install Python deps

```bash
git clone https://github.com/Nishant-ZFYII/nyc_hack.git
cd nyc_hack
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `nvidia-nat-langchain` may error on `pip`'s dependency resolver. If it does, run `pip install uv && uv pip install nvidia-nat-langchain` instead (uv handles deep dependency trees pip can't).

#### 4. Seed demo data

```bash
python3 seed_demo_cases.py
```

This writes 8 curated cases (1 critical, 2 high, 3 medium, 2 resolved) into `data/cases/` so the admin dashboard has something to show on first open.

#### 5. Launch both portals

Open **two terminals** (or run both in the background):

```bash
# Terminal 1 — user portal
uvicorn server:app --host 0.0.0.0 --port 9000

# Terminal 2 — admin portal
uvicorn admin_server:app --host 0.0.0.0 --port 9001
```

#### 6. Open in your browser

| Portal | URL | What to try |
|---|---|---|
| **Client / user** | [http://localhost:9000](http://localhost:9000) | Set location to *"Flatbush Brooklyn"* → type *"I have 4 kids and we're losing our housing next week"* → click **🌟 Not Sure Where to Start?** → wait ~60s for full plan → click **🎫 Raise a Ticket** to register the case |
| **Admin / caseworker** | [http://localhost:9001](http://localhost:9001) | Click any case → **📄 Fill Forms from ID** → upload `samples/sample_id.jpg` → download filled LDSS-4826 SNAP + DOH-4220 Medicaid PDFs |

#### Troubleshooting

| Symptom | Fix |
|---|---|
| Agent request hangs | Check Ollama is running: `curl http://localhost:11434/api/tags`. Restart with `OLLAMA_NUM_GPU=999 ollama serve &` |
| "tesseract not found" | Re-run step 1 — the fallback OCR needs the system binary even though we default to Ollama vision |
| `ModuleNotFoundError: No module named 'nat'` | `pip install nvidia-nat nvidia-nat-langchain` (use `uv pip install` if pip fails) |
| Admin page shows 0 cases | `python3 seed_demo_cases.py` (you skipped step 4) |
| Page shows only "Find Help" | Scroll down — the **🌟 Not Sure Where to Start?** button is the full-plan agent |

#### Stop everything

```bash
pkill -f uvicorn     # stops both portals
pkill ollama         # stops Ollama
```

---

## Architecture

```
User query → LLM Planner (Nemotron) → JSON Plan
  → Executor (cuDF/cuGraph/cuOpt on DGX)
  → Synthesizer (natural language answer)
  → Verifier (per-claim fact-checking)
  → Case tracker (persistent across visits)
  → User feedback loop (ground truth correction)
```

**Key principle:** The LLM never touches the data directly. It translates natural language to a structured JSON plan; the executor queries real data; the synthesizer formats the answer; the verifier fact-checks each claim against the mart.

## Current Status (April 12, 2026)

- ✅ FastAPI backend on port 9000 with custom HTML frontend (dark theme, deck.gl 3D map)
- ✅ Resource mart: **7,759 resources** across 19 types
- ✅ Knowledge graph: **3.6M edges** via cuGraph
- ✅ SPO triples: **328K + 1.2M** txt2kg extractions
- ✅ PyKEEN KGE: **83,618 entities × 64 dims** (TransE)
- ✅ cuOpt VRP allocation (cold emergency, migrant allocation)
- ✅ Location-aware search with GPS + manual address + Nominatim geocoding
- ✅ Multi-modal routing: walk (OSRM) + transit with budget awareness
- ✅ Case management: login, visit tracking, progress, choose/checkin/resolve
- ✅ Eligibility screener (SNAP/Medicaid/WIC/Cash/Fair Fares)
- ✅ Rights database (per resource type)
- ✅ Success stories (6 anonymized journeys)
- ✅ Ollama nemotron-mini on GPU (90 tok/s)

## Requirements

- Python 3.10+ (3.12 on DGX)
- DGX Spark (for GPU features) OR laptop with an LLM API key

### On DGX Spark (no API key needed)

```bash
# Start ollama on GPU (REQUIRED — default is CPU)
sudo systemctl stop ollama
OLLAMA_NUM_GPU=999 ollama serve &
ollama pull nemotron-mini  # or nemotron-3-nano for larger model

# Clone and install
git clone https://github.com/Nishant-ZFYII/nyc_hack.git
cd nyc_hack
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install NVIDIA GPU stack (aarch64 wheels for DGX Spark GB10)
pip install cudf-cu12 cugraph-cu12 cuml-cu12 cupy-cuda12x cuopt-cu12
pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps
pip install pykeen tensorrt-llm

# Run
uvicorn server:app --host 0.0.0.0 --port 9000
# Open http://<dgx-ip>:9000
```

### On laptop (API key required)

Full step-by-step for a fresh Linux/Mac laptop:

```bash
# 1. Clone the repo
git clone https://github.com/Nishant-ZFYII/nyc_hack.git
cd nyc_hack

# 2. Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
# .\venv\Scripts\activate     # Windows PowerShell

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Get an LLM API key
#    Option A: Anthropic Claude Haiku (recommended — fast, cheap, reliable JSON)
#      → https://console.anthropic.com/ → API keys → create key
#      → Add $5 credit
#    Option B: OpenAI
#      → https://platform.openai.com/api-keys

# 5. Run the server
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE uvicorn server:app --host 0.0.0.0 --port 9000

# 6. Open in browser
#    http://localhost:9000
```

**If port 9000 is busy:**
```bash
# Find what's using the port and kill it
lsof -ti:9000 | xargs kill -9
# Or run on a different port:
ANTHROPIC_API_KEY=sk-... uvicorn server:app --host 0.0.0.0 --port 9001
```

**To stop the server:** `Ctrl+C`

**To update code later:**
```bash
cd nyc_hack
source venv/bin/activate
git pull
pip install -r requirements.txt   # in case new deps were added
# restart uvicorn
```

**Persistent API key (don't type every time):**
```bash
# Linux/Mac — add to ~/.bashrc or ~/.zshrc
echo 'export ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE' >> ~/.bashrc
source ~/.bashrc
# Then just run:
uvicorn server:app --host 0.0.0.0 --port 9000
```

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'fastapi'` | `pip install -r requirements.txt` |
| `Address already in use` | `lsof -ti:9000 \| xargs kill -9` |
| `No LLM provider available` | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| Port 9000 works on DGX but not locally | Use `http://localhost:9000` not the DGX IP |
| `data/resource_mart.parquet not found` | `git pull` — data files are in the repo |
| Slow queries (>60s) | You're on CPU — expected. Faster on DGX with ollama |
| Permission errors on macOS | `sudo xcode-select --install` then retry pip install |

Data (`data/`, `stage/`) is **included in the repo** — no need to regenerate.

> **Alternate data source:** [Google Drive](https://drive.google.com/drive/folders/1gcCU1-kqGn64PfQ51Hw93J96nt-GI0br)

## API Endpoints

### Core Pipeline
- `POST /api/query` — main query. Body: `{query, case_id?, location?: {lat,lon}, demo_mode?}`. Returns: `{answer, plan, resources[], timing, verification, reasoning, clarify_question}`
- `GET /api/resources` — all 7,759 resources with coords (for map)
- `GET /api/status` — system health + resource counts

### Location & Routing
- `GET /api/geocode?q=<address>` — geocode an address via Nominatim proxy
- `POST /api/directions` — multi-modal directions. Body: `{from_lat, from_lon, to_lat, to_lon, budget?}`

### Case Management
- `POST /api/case/login` — create/resume case. Body: `{case_id, name?}`
- `GET /api/case/progress/{case_id}` — structured progress with ✅/🔄/⚪
- `POST /api/case/choose` — user picks a resource for a need. Body: `{case_id, need_category, resource_name, resource_address, resource_type}`
- `POST /api/case/checkin` — confirm arrival. Body: `{case_id, arrived: true/false, resource_name, feedback?}`
- `POST /api/case/resolve` — mark a need as resolved manually. Body: `{case_id, category}`
- `POST /api/case/visited` — log resource visit with feedback
- `GET /api/cases` — list all cases (admin view)

### Knowledge & Support
- `POST /api/eligibility` — benefits calculator. Body: `{household_size, annual_income, has_children, ...}`. Returns qualifying programs + estimated amounts.
- `GET /api/rights?resource_type=shelter` — legal rights at that resource type
- `GET /api/stories?need=housing&k=3` — anonymized success stories
- `POST /api/similar` — KGE similarity search. Body: `{resource_id, k}`
- `POST /api/feedback` — report issue with a resource. Body: `{resource_name, issue, detail}`

## Project Structure

```
nyc_hack/
├── server.py                 # FastAPI backend (main entry point)
├── frontend/
│   └── index.html            # Single-page app (dark theme, deck.gl map)
├── llm/
│   └── client.py             # LLM fallback ladder (ollama → Claude → OpenRouter)
├── pipeline/
│   ├── planner.py            # NL → JSON plan
│   ├── executor.py           # Plan → cuDF/cuGraph queries (with location-aware sorting)
│   ├── synth.py              # Results → natural language
│   ├── verify.py             # Per-claim fact verification
│   ├── feedback.py           # Ground truth correction
│   ├── clarify.py            # Multi-turn follow-up
│   ├── cases.py              # Case management (persistent user tracking)
│   ├── geocode.py            # Nominatim + NYC landmark fallback
│   ├── routing.py            # OSRM walking + MTA transit directions
│   └── eligibility.py        # Benefits calculator + rights + stories
├── engine/
│   ├── confidence.py         # Confidence-scored reasoning paths
│   ├── embeddings.py         # KGE embeddings (loads PyKEEN if available)
│   ├── txt2kg.py             # 311 complaint → structured triples
│   └── train_kge.py          # Train PyKEEN TransE on triples
├── data/                     # INCLUDED: resource_mart, graph.pkl, triples, KGE embeddings
├── stage/                    # INCLUDED: cleaned per-dataset parquet
├── pull_all.py               # Download raw data from NYC Open Data
├── clean_all.py              # Clean raw data
├── build_mart.py             # Build unified resource mart
├── build_graph.py            # Build cuGraph knowledge graph
├── build_triples.py          # Build SPO triples
├── ARCHITECTURE.md           # Detailed technical docs
├── DEMO_SCRIPT.md            # Demo presentation
└── DEVPOST.md                # Hackathon submission writeup
```

## Example Queries

**Needs assessment (caseworker):**
- *"I'm Tina, 4 kids ages 12-16, income $28K, my sister is kicking us out next week"*

**Simple lookup:**
- *"What shelters in Brooklyn have available beds?"*

**Location-aware:**
- *"I need a shelter near 43rd street Manhattan"*

**City-ops simulation:**
- *"Cold emergency declared. 3 Brooklyn shelters hit capacity. 200 people outside at 15°F. What do we do?"*

## LLM Provider Fallback (`llm/client.py`)

1. Ollama nemotron-mini at `localhost:11434` (primary on DGX)
2. Claude Haiku via Anthropic API
3. GPT-4o-mini via OpenAI API
4. llama.cpp at `localhost:8080`
5. OpenRouter (last resort)

## NVIDIA Stack (11 components)

| Tool | Role |
|------|------|
| **Nemotron via Ollama** | LLM inference on GB10 GPU |
| **RAPIDS cuDF** | GPU DataFrame filtering |
| **RAPIDS cuGraph** | 3.6M-edge knowledge graph |
| **cuOpt** | VRP allocation for emergencies |
| **cuPy** | GPU spatial scoring |
| **RAPIDS cuML** | KNN / HDBSCAN |
| **TensorRT-LLM** | Compiled inference |
| **PyKEEN + PyTorch** | TransE KGE embeddings |
| **txt2kg** | 311 complaint extraction |
| **deck.gl** | GPU 3D map |
| **CUDA 13.0** | DGX Spark GB10 compute |

## License

MIT
