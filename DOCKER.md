# Running this project (for judges)

**Three commands. Two URLs. Done.**

## Prerequisites

- Docker + Docker Compose v2+ (`docker compose version`)
- **Recommended:** NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for fast inference
- **~30 GB free disk** for the two LLMs Ollama will download (llama3, llama3.2-vision)
- ~10 GB free RAM (more if not using GPU)

If you don't have a GPU, open `docker-compose.yml` and delete the `deploy.resources.reservations.devices` block under the `ollama` service. Everything still runs — just slower.

## Start

**Recommended** — use the smart launcher which auto-detects an existing host Ollama cache so you don't re-download 25 GB of models:

```bash
git clone https://github.com/Nishant-ZFYII/nyc_hack.git && cd nyc_hack
./start.sh
```

**Manual equivalent** if you prefer not to use the script:

```bash
git clone https://github.com/Nishant-ZFYII/nyc_hack.git && cd nyc_hack
docker compose up -d
docker compose logs -f ollama-init       # Ctrl-C once "Models ready." appears
```

When `ollama-init` prints `Models ready.` and exits, both portals are live.

## Use

| Portal | URL | What to try |
|---|---|---|
| **Client / user** | [http://localhost:9000](http://localhost:9000) | Set location to *"Flatbush Brooklyn"* → type *"I need a shelter tonight"* → click **🌟 Not Sure Where to Start?** → wait ~60 s for the action plan → click **🎫 Raise a Ticket** to register a case |
| **Admin / caseworker** | [http://localhost:9001](http://localhost:9001) | See 8 pre-seeded cases by urgency → click a case → **📄 Fill Forms from ID** → upload `samples/sample_id.jpg` → download filled LDSS-4826 + DOH-4220 PDFs |

## Stop

```bash
docker compose down         # stops containers, keeps model cache
docker compose down -v      # also removes model cache — forces re-download next time
```

## Architecture at a glance

```
┌───────────────┐   ┌────────────────┐   ┌─────────────────┐
│ user portal   │   │ admin portal   │   │ ollama service  │
│ :9000         │←──│ :9001          │←──│ :11434 (GPU)    │
│ FastAPI       │   │ FastAPI        │   │ llama3 +        │
│ NeMo ReAct    │   │ AI briefing +  │   │ llama3.2-vision │
│ agent         │   │ form filler    │   │                 │
└───────────────┘   └────────────────┘   └─────────────────┘
       │                    │                      ▲
       │                    │                      │
       └────────────────────┴──────────────────────┘
                  shared ./data/cases volume
                  (JSON case files, lives on host)
```

- **NVIDIA NeMo Agent Toolkit** drives the ReAct loop for both portals
- **NVIDIA NeMo Guardrails** gates every LLM call (blocks PII leaks, jailbreaks, crisis keywords)
- **llama3** reasons over Python tools (`find_resources`, `calculate_eligibility`, `get_rights`, `get_directions`, etc.)
- **llama3.2-vision** reads ID photos for the form-filling feature
- All local, no cloud

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker compose up` says no GPU | Install NVIDIA Container Toolkit, or delete the GPU block in compose and accept CPU speed |
| Pages load but agent times out | `docker compose logs ollama-init` — make sure models finished downloading |
| "Case list is empty" | `docker compose exec app-user python3 seed_demo_cases.py` to seed 8 demo cases |
| Port 9000/9001 already in use | Change the left-hand port in `ports:` lines in `docker-compose.yml` |
| `llama3.2-vision:11b` pull fails | Re-run `docker compose exec ollama-init ollama pull llama3.2-vision:11b` |

## Seed data

Fresh containers come up empty. The `app-user` service auto-runs `seed_demo_cases.py --seed-only` on first boot, which writes 8 curated cases (1 critical DV, 2 high, 3 medium, 2 resolved) to `./data/cases/`. You can re-seed anytime:

```bash
docker compose exec app-user python3 seed_demo_cases.py --no-backup
```

## What to look at if you're evaluating

- **`agent/register.py`** — NeMo Agent Toolkit tool groups (5 groups, 15+ tools)
- **`agent/config.yml`** + **`agent/config_admin.yml`** — ReAct agent configs
- **`pipeline/form_filler.py`** — llama3.2-vision OCR + real NYC PDF overlay
- **`guardrails/`** — actual `nemoguardrails` integration (flows.co + config.yml)
- **`server.py`** — user portal endpoints
- **`admin_server.py`** — admin portal endpoints
- **`frontend/index.html`** — single-file SPA for users
- **`frontend/admin.html`** — single-file SPA for caseworkers
