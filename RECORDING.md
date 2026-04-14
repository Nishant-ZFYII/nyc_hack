# LinkedIn Recording — Quick Start

Everything below assumes you're on the local laptop at `/home/nishant/nyc_hack/`, Python env `nyc_hack`, GTX 1060 + 16 GB RAM. No DGX, no cloud.

## 1. Pre-flight (once, ~5 min)

```bash
# Conda env with all deps — already installed this session
conda activate nyc_hack

# Ollama with llama3 (for the agent drawer only; optional for hero reel)
ollama pull llama3
OLLAMA_KEEP_ALIVE=2h ollama serve > /tmp/ollama.log 2>&1 &
```

## 2. Start the two portals

```bash
cd /home/nishant/nyc_hack
# User portal :9000  (hero 3D scenario loop)
uvicorn server:app --host 127.0.0.1 --port 9000 --workers 1 --loop asyncio > /tmp/server.log 2>&1 &
# Admin portal :9001 (caseload ops map)
uvicorn admin_server:app --host 127.0.0.1 --port 9001 --workers 1 --loop asyncio > /tmp/admin.log 2>&1 &

# Smoke test
curl -s http://127.0.0.1:9000/api/resources | head -c 100; echo
curl -s http://127.0.0.1:9001/api/admin/ops_snapshot | head -c 100; echo
```

(Seed demo cases first if admin is empty: `python3 seed_demo_cases.py --no-backup`.)

## 3. Browser setup (Chrome recommended)

- **Chrome flags:** `chrome --use-gl=desktop --enable-gpu-rasterization` (deck.gl WebGL2)
- **Zoom:** 100 %
- **Close:** Slack, Discord, DevTools
- **Do Not Disturb** on so no notifications appear mid-take
- **Hide the bookmarks bar** (`Ctrl+Shift+B`)

Two tabs:
- Tab 1: `http://localhost:9000/` → click **Live Ops Demo** card → auto-loop starts
- Tab 2: `http://localhost:9001/` → **Dashboard** → click **📺 Live Ops Map** in topbar

## 4. Recording

Using **OBS Studio** (any similar works):
- Canvas + Output: **1920×1080 @ 60 fps**
- Encoder: **NVENC H.264**, CQP 18, keyframe 2 s
- Audio: **muted** — LinkedIn autoplay is silent anyway
- Scene: capture the Chrome window fullscreen (F11)

**Takes:**
1. User side — `http://localhost:9000/` → `Live Ops Demo`. Let it run through **one full scenario cycle** (~25 s: cold snap → migrant bus → reset). Camera auto-orbits the whole time.
2. Admin side — `http://localhost:9001/` → `Live Ops Map`. ~15 s hovering over the 3D city with cases pulsing + arcs flowing, HUD counters visible.
3. **Optional 3rd take:** user page, open the agent drawer (🤖 NEMO AGENT button, left), click the canned query, show the LLM response.

Do 2 takes per shot, pick the cleanest.

## 5. Edit (kdenlive / shotcut / iMovie)

Suggested 30-second cut:
```
0–2s   title card: "NYC SOCIAL SERVICES · LIVE OPS · RUNNING LOCALLY"
2–14s  user hero loop (cold emergency arcs → migrant bus)
14–24s admin ops map (pulsing cases, load-colored columns)
24–28s optional: agent drawer LLM moment
28–30s logo strip: "NVIDIA NeMo Agent Toolkit · deck.gl · llama3 · CPU-only laptop"
```

Export: **MP4 H.264, 8 Mbps, 1080 p, AAC muted** — target under 200 MB for LinkedIn.

Verify the file plays on mobile Chrome before uploading.

## 6. LinkedIn post template

```
Rebuilt my NVIDIA Spark Hack NYC project for my laptop.

Same engineering:
- 7,759 NYC social services mapped with deck.gl 3D columns
- Pure-Python scenario sim (cold emergency, migrant bus) — no GPU RAPIDS,
  no DGX, no cloud
- NVIDIA NeMo Agent Toolkit drawer with llama3 on Ollama
- Dark-matter basemap, auto-orbiting camera, live stats HUD
- Runs on a GTX 1060 6 GB

The win from watching what did get winners: visually immediate + zero
LLM on the critical path. The LLM is the drawer, not the demo.

Code: github.com/Nishant-ZFYII/nyc_hack
#NVIDIA #deckgl #NYC #PublicInterestAI
```

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Live Ops map is flat / dim | Wait 3 s after opening — resources load async; columns rise after first fetch |
| Arcs don't appear | Open DevTools → Network → check `/api/scenario/cold_emergency` returns 200 |
| Auto-orbit paused | You clicked / scrolled — orbit resumes after 5 s idle |
| Agent drawer shows "LLM offline" | `ollama run llama3` to pull + warm the model |
| Admin map is empty | `python3 seed_demo_cases.py --no-backup` to seed 9 demo cases |
| GPU hot, fps dropping | Close other tabs; Chrome task manager → end sibling tabs |
| Port in use | `ss -tlnp \| grep -E ':9000\|:9001'` → `kill -9 <pid>` |

## 8. Stop

```bash
pkill -f uvicorn
pkill ollama
```
