# NYC Social Services Intelligence Engine

An AI-powered caseworker + kiosk system for NYC Department of Social Services. Built for **NVIDIA Spark Hack NYC 2026**.

The system performs comprehensive needs assessments across housing, food, healthcare, legal aid, benefits eligibility, school continuity, and safety — using a **bounded DSL architecture** that prevents LLM hallucination. It tracks people across multiple visits (like a real caseworker), finds resources sorted by distance from where they are, and guides them step-by-step through their journey.

---

## 🏁 Quickstart for evaluators (judges read this)

Pick ONE path below. **Path A (Docker)** is the zero-assumption path — works on any machine with Docker + internet. **Path B (bare metal)** is for developers who want to run directly.

### Path A — Docker (recommended for judges)

**Prerequisites** (all standard on modern dev machines):

```bash
docker --version           # need v24+
docker compose version     # need v2+
```

If either is missing:
- **Ubuntu/Debian:** `sudo apt-get install -y docker.io docker-compose-plugin && sudo usermod -aG docker $USER && newgrp docker`
- **Mac:** install [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/)
- **Windows:** install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) + enable WSL2

**Optional but recommended:** if you have an NVIDIA GPU, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Without a GPU, Ollama falls back to CPU (still works, just slower).

Disk + RAM: **~30 GB free disk** (for the two models), **~10 GB free RAM** minimum.

**Run it:**

```bash
# 1. Clone
git clone https://github.com/Nishant-ZFYII/nyc_hack.git
cd nyc_hack

# 2. Launch (smart — reuses existing Ollama cache if present)
./start.sh
```

The `start.sh` script does the full boot:
- Auto-detects an existing host Ollama cache (`~/.ollama`) and bind-mounts it so you don't re-download 25 GB
- If no cache, Docker downloads `llama3` + `llama3.2-vision:11b` on first run (10-20 min)
- Boots the stack (user portal + admin portal + ollama)
- Tails the init container until models are ready
- Prints the two URLs when done

**If you don't want the helper script**, the manual equivalent is:

```bash
docker compose up -d
docker compose logs -f ollama-init       # wait for "Models ready."
```

**Open in your browser:**

| Portal | URL | First thing to try |
|---|---|---|
| **Client / user** | [http://localhost:9000](http://localhost:9000) | Set location to *"Flatbush Brooklyn"* → type *"I have 4 kids and we're losing our housing next week"* → click **🌟 Not Sure Where to Start?** → wait ~60s → click **🎫 Raise a Ticket** to register the case |
| **Admin / caseworker** | [http://localhost:9001](http://localhost:9001) | You'll see 8 pre-seeded cases → click any case → **📄 Fill Forms from ID** → upload `samples/sample_id.jpg` → download filled LDSS-4826 SNAP + DOH-4220 Medicaid PDFs |

**Stop the stack:**
```bash
./start.sh stop              # or: docker compose down
./start.sh reset             # wipe Docker volumes + regenerated override (keeps host Ollama cache)
```

See [DOCKER.md](./DOCKER.md) for container architecture details.

---

### Path B — Bare metal (no Docker)

Works on Linux/macOS. Tested on Ubuntu 22.04 and the NVIDIA DGX Spark.

#### 1. System prerequisites

```bash
# Python 3.10 or newer (3.12 recommended)
python3 --version                         # must print 3.10+

# Ubuntu/Debian — full system deps (OCR, C++ compiler for annoy, image libs, curl)
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    build-essential g++ python3-dev \
    libgl1 libglib2.0-0 \
    curl

# macOS — Xcode tools (provides clang++) + Tesseract via Homebrew
xcode-select --install 2>/dev/null || true
brew install tesseract
```

> **Why `build-essential`?** `nemoguardrails` transitively requires `annoy`, which compiles native C++ code. Without `g++` the `pip install` step will fail with `command 'g++' failed: No such file or directory`.

#### 2. Install Ollama + pull the two models

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama on GPU (omit env vars if you don't have a GPU)
sudo systemctl stop ollama 2>/dev/null
OLLAMA_NUM_GPU=999 OLLAMA_KEEP_ALIVE=2h ollama serve > /tmp/ollama.log 2>&1 &

# Pull the two models (~25 GB total, 10-20 min first run)
ollama pull llama3                        # ReAct agent (~4.7 GB)
ollama pull llama3.2-vision:11b           # ID reader (~7.9 GB)

# Verify — should list both models
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

#### 3. Clone + install Python deps

```bash
git clone https://github.com/Nishant-ZFYII/nyc_hack.git
cd nyc_hack
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip uv        # uv handles our deep dep graph (nvidia-nat + langchain + cuda stack)
uv pip install -r requirements.txt
```

> **Why `uv` not plain `pip`?** `nvidia-nat-langchain` + the cuda/torch stack pulls ~200 transitive deps. Plain pip will error with `resolution-too-deep`. `uv` is a drop-in pip replacement that handles it in ~60 seconds.
>
> If you must use pip: `pip install --use-deprecated=legacy-resolver -r requirements.txt` (slower, may have version conflicts).

#### 4. Seed demo data

```bash
python3 seed_demo_cases.py
```

Writes 8 curated cases (1 critical DV, 2 high, 3 medium, 2 resolved) into `data/cases/` so the admin dashboard has demo content. The resource mart, knowledge graph, and embeddings are already in `data/` (committed to git — no regeneration needed).

#### 5. Launch both portals

Open **two terminals**:

```bash
# Terminal 1 — user portal
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 9000

# Terminal 2 — admin portal
source venv/bin/activate
uvicorn admin_server:app --host 0.0.0.0 --port 9001
```

#### 6. Open in browser

Same URLs and flows as Path A above. Use `localhost` or your machine's IP.

#### Troubleshooting (bare metal)

| Symptom | Fix |
|---|---|
| Agent request hangs forever | `curl http://localhost:11434/api/tags` — if no response, restart Ollama: `pkill ollama; OLLAMA_NUM_GPU=999 ollama serve &` |
| `tesseract: command not found` | Re-run step 1 — fallback OCR needs the system binary |
| `ModuleNotFoundError: No module named 'nat'` | `pip install nvidia-nat nvidia-nat-langchain` (use `uv pip install` if pip fails) |
| Admin page shows 0 cases | You skipped step 4 — run `python3 seed_demo_cases.py` |
| Only `Find Help` button, no AI plan button | Scroll down — the `🌟 Not Sure Where to Start?` button is below it |
| `data/resource_mart.parquet not found` | `git pull` — data files are committed to git |
| Port 9000 or 9001 already in use | `lsof -ti:9000 \| xargs kill -9` or change the port flag |
| Slow queries (>2 min) | You're on CPU — expected. Faster with GPU ollama |

#### Stop everything

```bash
pkill -f uvicorn           # both portals
pkill ollama               # Ollama
```

---

## Architecture

```
User query → LLM Planner (llama3) → JSON Plan
  → Executor (cuDF/cuGraph/cuOpt on DGX)
  → Synthesizer (natural language answer)
  → Verifier (per-claim fact-checking)
  → Case tracker (persistent across visits)
  → User feedback loop (ground truth correction)
```

**Key principle:** the LLM never touches the data directly. It translates natural language to a structured JSON plan; the executor queries real data; the synthesizer formats the answer; the verifier fact-checks each claim against the resource mart.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

## Current status (April 12, 2026)

- ✅ FastAPI user portal on `:9000` + admin portal on `:9001`
- ✅ **NVIDIA NeMo Agent Toolkit** — ReAct agent on llama3, 5 tool groups, 15+ tools (`agent/register.py`)
- ✅ **NVIDIA NeMo Guardrails** — PII detection, jailbreak blocking, crisis keyword handling (`guardrails/`)
- ✅ **OpenClaw skill** — `nyc-caseworker` capsule submitted to ClawHub (`skills/nyc-caseworker/`)
- ✅ Resource mart: **7,759 resources** across 19 types
- ✅ Knowledge graph: **3.6M edges** via cuGraph
- ✅ SPO triples: **328K + 1.2M** txt2kg extractions
- ✅ PyKEEN KGE: **83,618 entities × 64 dims** (TransE)
- ✅ cuOpt VRP allocation (cold emergency, migrant allocation)
- ✅ Location-aware search with GPS + address geocoding (Nominatim proxy + fallback table)
- ✅ Multi-modal routing: walk (OSRM) + transit with budget awareness + sponsored ride
- ✅ Case management: login, visit tracking, progress, choose/checkin/resolve
- ✅ Eligibility screener (SNAP/Medicaid/WIC/Cash/Fair Fares + auto-fill)
- ✅ Rights database (per resource type)
- ✅ llama3.2-vision multimodal OCR → auto-filled NYC benefits PDFs
- ✅ Success stories (10+ anonymized journeys)

## API endpoints

### Core pipeline
- `POST /api/query` — main query. Body: `{query, case_id?, location?: {lat,lon}, demo_mode?}`
- `POST /api/agent/plan` — autonomous plan (timeline view with alternatives, eligibility, rights, stories)
- `POST /api/agent/nat` — NeMo Agent Toolkit ReAct answer
- `POST /api/agent/openclaw` — OpenClaw skill dispatch
- `GET /api/resources` — all 7,759 resources with coords (for map)
- `GET /api/status` — system health + resource counts

### Location & routing
- `GET /api/geocode?q=<address>` — Nominatim proxy + NYC fallback table
- `POST /api/directions` — multi-modal directions. Body: `{from_lat, from_lon, to_lat, to_lon, budget?}`

### Case management
- `POST /api/case/login` — create/resume case
- `GET /api/case/progress/{case_id}` — structured progress (✅ / 🔄 / ⚪)
- `POST /api/case/choose` — user picks a resource for a need
- `POST /api/case/checkin` — confirm arrival (or not)
- `POST /api/case/resolve` — mark a need as resolved
- `POST /api/ticket/raise` — register a formal NYC case ticket (NYC-XXXXXX)

### Knowledge & support
- `POST /api/eligibility` — benefits calculator
- `GET /api/rights?resource_type=shelter` — legal rights
- `GET /api/stories?need=housing&k=3` — success stories
- `POST /api/refine` — checkbox-based refine panel (updates eligibility + rights + resources)
- `POST /api/similar` — KGE similarity search
- `POST /api/feedback` — report issue with a resource

### Admin
- `GET /api/admin/cases` — list all cases
- `POST /api/admin/fill_forms` — auto-fill NYC benefits PDFs from ID upload
- `POST /api/admin/ocr_id` — llama3.2-vision OCR on ID photo

## Project structure

```
nyc_hack/
├── server.py                     # user portal (:9000)
├── admin_server.py               # admin portal (:9001)
├── frontend/
│   ├── index.html                # user SPA (dark→light theme, deck.gl map, emoji icons)
│   └── admin.html                # caseworker SPA
├── agent/                        # NVIDIA NeMo Agent Toolkit integration
│   ├── register.py               # tool groups + skills
│   └── config.yml                # ReAct agent config
├── skills/nyc-caseworker/        # OpenClaw skill capsule for ClawHub
├── guardrails/                   # NeMo Guardrails (PII, jailbreak, crisis)
├── llm/client.py                 # LLM fallback ladder
├── pipeline/
│   ├── planner.py                # NL → JSON plan
│   ├── executor.py               # plan → cuDF/cuGraph queries
│   ├── synth.py                  # results → natural language
│   ├── verify.py                 # per-claim fact check
│   ├── cases.py                  # case management
│   ├── geocode.py                # Nominatim + fallback table
│   ├── routing.py                # OSRM walking + MTA transit
│   ├── eligibility.py            # benefits calculator
│   ├── form_filler.py            # llama3.2-vision OCR + PDF overlay
│   └── briefing.py               # admin briefings
├── engine/                       # KGE + cuGraph primitives
├── data/                         # ✅ committed: mart, graph.pkl, triples, KGE
├── samples/                      # sample ID image + blank NYC forms
├── seed_demo_cases.py            # writes 8 demo cases to data/cases/
├── docker-compose.yml            # full stack orchestration
├── Dockerfile                    # app image
├── start.sh                      # smart launcher (auto-detect ollama cache)
├── ARCHITECTURE.md
├── DOCKER.md
├── DEMO_SCRIPT.md
└── DEVPOST.md                    # hackathon submission
```

## Example queries

**Caseworker-style (rich needs assessment):**
- *"I'm Tina, 4 kids ages 12-16, income $28K, my sister is kicking us out next week"*
- *"Someone broke into my apartment, I have a 6-year-old, I don't feel safe going back"*

**Simple lookup:**
- *"What shelters in Brooklyn have available beds?"*
- *"Free food pantries near Jamaica Queens"*

**Location-aware:**
- *"I need a shelter near 43rd street Manhattan"*

**City-ops simulation:**
- *"Cold emergency declared. 3 Brooklyn shelters hit capacity. 200 people outside at 15°F. What do we do?"*

## LLM provider fallback (`llm/client.py`)

1. **Ollama llama3 on DGX** (primary, local)
2. Ollama nemotron-mini (secondary, local)
3. Claude Haiku via Anthropic API
4. GPT-4o-mini via OpenAI API
5. llama.cpp at `localhost:8080`
6. OpenRouter (last resort)

## NVIDIA stack (11+ components)

| Tool | Role |
|------|------|
| **Nemotron / llama3 via Ollama** | Local LLM inference on GB10 GPU |
| **NeMo Agent Toolkit** | ReAct agent + tool dispatch |
| **NeMo Guardrails** | PII / jailbreak / crisis detection |
| **OpenClaw + ClawHub** | `nyc-caseworker` skill capsule |
| **RAPIDS cuDF** | GPU DataFrame filtering |
| **RAPIDS cuGraph** | 3.6M-edge knowledge graph |
| **cuOpt** | VRP allocation for emergencies |
| **cuPy** | GPU spatial scoring |
| **RAPIDS cuML** | KNN / HDBSCAN |
| **PyKEEN + PyTorch** | TransE KGE embeddings |
| **txt2kg** | 311 complaint extraction |
| **deck.gl** | GPU 3D map (IconLayer + ScatterplotLayer) |

## License

MIT
