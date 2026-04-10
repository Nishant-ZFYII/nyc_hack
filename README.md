# NYC Social Services Intelligence Engine

An AI-powered caseworker system for NYC Department of Social Services. Built for NVIDIA Spark Hack NYC (April 10-12, 2026).

The system performs comprehensive needs assessments — decomposing life situations into prioritized needs across housing, food, healthcare, legal aid, benefits eligibility, and safety — using a bounded DSL architecture that prevents hallucination.

## Architecture

```
User query → LLM Planner (Nemotron/Claude) → JSON Plan
  → Executor (cuDF/cuGraph on DGX, pandas/networkx locally)
  → Synthesizer (natural language answer)
  → Verifier (per-claim fact-checking)
  → User feedback loop (ground truth correction)
```

**Key principle:** The LLM never answers directly. It produces a structured JSON plan; the executor queries real data; the synthesizer formats the answer; the verifier fact-checks each claim.

## Requirements

- Python 3.10+
- An LLM API key (at least one):
  - `ANTHROPIC_API_KEY` — Claude Haiku (recommended for local dev)
  - `OPENAI_API_KEY` — GPT-4o-mini
  - `OPENROUTER_API_KEY` — OpenRouter (free Nemotron tier)
  - Or a local NIM/vLLM server on `localhost:8000`

**Yes, an API key is required.** The system uses an LLM for query planning, answer synthesis, verification, and feedback parsing. Without an LLM provider, it cannot function.

### Install dependencies

```bash
pip install streamlit pandas numpy networkx pyarrow requests openai anthropic scikit-learn
```

## Setup & Run

### Step 1: Pull raw data from NYC Open Data

```bash
python pull_all.py
```

This downloads ~10 datasets from NYC Open Data (Socrata API) into `raw/`. Takes ~5-10 minutes depending on connection.

### Step 2: Clean and normalize

```bash
python clean_all.py
```

Processes raw data into standardized parquet files in `stage/`. Handles coordinate conversions, schema normalization, and deduplication.

### Step 3: Build the resource mart

```bash
python build_mart.py
```

Combines all staged datasets into a unified `data/resource_mart.parquet` (~7,700 resources across 19 types) with safety scores, quality scores, and spatial attributes.

### Step 4: Build the knowledge graph

```bash
python build_graph.py
```

Constructs a NetworkX graph with resource nodes, transit stations, census tracts, and edges (NEAR, WALK_TO_TRANSIT, IN_TRACT, TRANSIT_LINK). Saved as `data/graph.pkl`.

### Step 5 (optional): Build SPO triples and txt2kg

```bash
python build_triples.py
```

Generates subject-predicate-object triples for the knowledge graph and 311 complaint extraction.

### Step 6: Run the app

```bash
ANTHROPIC_API_KEY=your-key-here streamlit run app.py
```

Or with OpenAI:
```bash
OPENAI_API_KEY=your-key-here streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Quick Start (if you have pre-built data)

If someone shares the `data/` directory with you (contains `resource_mart.parquet` and `graph.pkl`), you can skip steps 1-5:

```bash
# Just copy data/ into the project root, then:
ANTHROPIC_API_KEY=your-key-here streamlit run app.py
```

## Project Structure

```
nyc_hack/
├── app.py                  # Streamlit UI (main entry point)
├── llm/
│   └── client.py           # LLM provider fallback ladder (NIM → Claude → GPT → OpenRouter)
├── pipeline/
│   ├── planner.py          # NL → JSON plan (needs_assessment, lookup, simulate, explain)
│   ├── executor.py         # Execute plan against resource mart + graph
│   ├── synth.py            # Synthesize natural language answer
│   ├── verify.py           # Per-claim fact verification + reasoning paths
│   ├── feedback.py         # User-in-the-loop ground truth correction
│   └── clarify.py          # Multi-turn follow-up questions
├── engine/
│   ├── confidence.py       # Confidence-scored reasoning paths
│   ├── embeddings.py       # KGE embeddings for similarity search
│   └── txt2kg.py           # 311 complaint → structured triples
├── pull_all.py             # Download raw data from NYC Open Data
├── clean_all.py            # Clean and normalize raw data
├── build_mart.py           # Build unified resource mart
├── build_graph.py          # Build knowledge graph
├── build_triples.py        # Build SPO triples
├── ARCHITECTURE.md         # Detailed technical documentation
├── DEMO_SCRIPT.md          # Demo presentation script
└── DEVPOST.md              # Hackathon submission writeup
```

## Example Queries

**Caseworker needs assessment:**
- "I'm Tina, 4 kids ages 12-16, income $28K, currently staying with my sister in Flatbush but she's kicking me out next week"
- "Someone broke into my apartment and stole my stuff. I don't feel safe going back. I have a 6 year old."

**Simple resource lookup:**
- "What shelters in Manhattan have available beds?"
- "Find wheelchair-accessible hospitals in Queens"

**City-ops simulation:**
- "A cold emergency is declared. 3 Brooklyn shelters hit capacity. 200 people outside. 15 degrees. What do we do?"
- "Which neighborhoods are more than 30 minutes from the nearest shelter?"

## LLM Provider Fallback

The system tries providers in order:
1. NIM container on DGX Spark (localhost:8000)
2. Claude Haiku via Anthropic API
3. GPT-4o-mini via OpenAI API
4. vLLM on DGX Spark (localhost:8001)
5. llama.cpp local server (localhost:8080)
6. OpenRouter Nemotron (free tier)

Set the appropriate environment variable for whichever provider you want to use.

## NVIDIA Stack (on DGX Spark)

On the DGX Spark, the system uses GPU-accelerated equivalents:
- **cuDF** replaces pandas for filtering
- **cuGraph** replaces networkx for graph operations
- **cuOpt** for VRP-based resource allocation
- **NIM** for local Nemotron-3-Nano-30B inference
- **cuML** for KNN spatial joins
- **cuPy** for accessibility heatmaps

Locally, everything runs on CPU with pandas/networkx/scikit-learn.

## License

MIT
