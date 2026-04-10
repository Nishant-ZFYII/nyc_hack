# Demo Script — NYC Social Services Intelligence Engine
## NVIDIA Spark Hack NYC · April 10–12, 2026

**Total demo time: 8–10 minutes**
**One sentence pitch:** "An AI system that does in 10 seconds what a human caseworker takes 45 minutes — a full needs assessment for any life situation a New Yorker walks in with."

---

## Opening (30 seconds)

> "NYC has 33,000 DHS caseworkers. Each one spends 45 minutes per client doing what a good caseworker does — figuring out not just what you asked for, but everything you need and didn't know to ask for. Housing, food, benefits, healthcare, legal rights, school continuity.
>
> We built an AI caseworker that does that in 10 seconds. And it runs entirely on this — a 1.5 kg workstation with 128 GB of unified memory and a GB10 GPU. No cloud. No privacy risk. 8 million New Yorkers' most sensitive data never leaves the building."

---

## Demo 1 — Caseworker Needs Assessment: Tina (2.5 min)

**[Click: 🏠 Caseworker — Family at risk]**

> "Let's start with the hardest case. Tina has 4 teenagers, makes $28K a year, and her sister is evicting her from a Flatbush apartment next week. She walks into a DHS office. What does a good caseworker do?"

**[Point to the Intent badge: 🏥 needs_assessment]**

> "Nemotron — running locally on the DGX, no internet — reads this situation and does what a trained social worker does. It identifies three separate needs: immediate housing, benefits enrollment — because at $28K with 4 kids she qualifies for SNAP and Medicaid — and school continuity for the teenagers."

**[Point to the resource tabs]**

> "The executor runs three separate searches against our knowledge graph — 7,759 resources across 19 types — and returns shelter options sorted by distance from Flatbush, benefits centers in Brooklyn, and schools nearby."

**[Point to the Answer section]**

> "And the synthesizer writes what the caseworker would actually say to Tina. Named resources, real addresses, next steps. Not 'visit mybenefits.ny.gov.' Specific."

**[If clarifying question appears]**

> "Notice it asked a follow-up. The system knows it doesn't have enough information yet — disability status would change which shelters are appropriate. This is the multi-turn caseworker loop."

---

## Demo 2 — Cold Emergency Simulation (2.5 min) ⭐ HERO DEMO

**[Click: ❄️ Sim — Cold emergency]**

> "Now switch roles. You're not a caseworker — you're an emergency coordinator at the NYC Office of Emergency Management. It's 15°F. Three Brooklyn shelters just hit capacity. 200 people are still outside."

**[Watch it run — point to the spinner]**

> "What's happening right now: cuDF is filtering 857,000 PLUTO tax lots — every building in New York City — for assembly-zoned structures. Churches, community centers, school gyms. Things that can hold 200 people tonight. cuGraph is computing distance from the displaced population to each candidate site. And Nemotron is synthesizing the response."

**[Point to the overflow sites table]**

> "These aren't shelters. These are buildings identified from NYC's tax database — PLUTO — that have assembly zoning. The Roman Catholic Diocese of Brooklyn. Coptic Orthodox Church. Congregation Khal Chasidei Skwera. These are real buildings that can hold people tonight, that a human coordinator might not know to call."

**[Point to the map expander]**

> "All of this — the shelter locations, the overflow sites, the food distribution points — plotted on a map in real time."

> "On a laptop, this query takes 3 minutes. On the DGX with cuDF and cuGraph running in unified memory, it takes 3 seconds. That's the difference between a coordinator making a decision and a coordinator watching a spinner."

---

## Demo 3 — Resource Gap (1 min)

**[Click: 📊 Sim — Resource gap]**

> "Equity question. Which boroughs are most underserved per capita? The Bronx has 1.5 million people. How many shelter, food bank, and hospital resources per 100,000 residents does it have versus Manhattan?"

**[Point to the results table]**

> "This is the kind of analysis that normally takes a policy analyst a week. We run it in under a second because the entire resource mart and population data live in cuDF in memory."

---

## Demo 4 — Migrant Allocation (1 min)

**[Click: 🌍 Sim — Migrant allocation]**

> "Last scenario. A bus just arrived with 80 people. They speak Spanish and Mandarin. They need shelter tonight, food tomorrow, and schools for the children."

**[Point to the allocation table]**

> "The system filters for shelters with language services, distributes 80 people across available sites, and co-locates food banks and schools in the same neighborhoods. This is a vehicle routing problem — the kind cuOpt was built for."

---

## Technical Close (1 min)

> "Let me show you what's in memory right now."

**[Point to sidebar: Mart: 7,759 resources · Graph: nodes · edges]**

> "Seven thousand seven hundred fifty-nine resources. 857,000 PLUTO lots. A knowledge graph with 120,000 edges. Nemotron-3-Nano-30B. All of it in unified memory. No swapping. No cloud calls. Every query under 3 seconds.
>
> The full production stack — with 311 history, NYPD crime data for safety routing, MTA real-time delays, and ACS demographics — uses about 31 gigabytes. This machine has 128. We're not close to the ceiling.
>
> NYC has 8 million residents. 33,000 caseworkers. This gives each one of them superhuman knowledge of every resource in the city. And it runs on hardware you can carry onto a subway."

---

## Backup answers for judge questions

**"Why can't this run on a laptop?"**
> "The full dataset — PLUTO, 311 history, NYPD data, MTA graph, Nemotron weights — is 31 GB. A laptop has 16 GB RAM shared with the OS. You'd be swapping constantly. The DGX's unified memory means the GPU and CPU share the same 128 GB — no copies, no transfers, the graph lives where the compute is."

**"Why not just use GPT-4 via API?"**
> "Two reasons. Privacy — this is the most sensitive data in government. A DHS caseworker cannot send a client's situation to OpenAI. And latency — API round trips add 1-3 seconds per call. We make 3 LLM calls per query. On the DGX, Nemotron runs locally at under 500ms per call."

**"How do you prevent hallucination?"**
> "Bounded DSL. Nemotron never answers questions directly. It translates natural language into a JSON plan — needs_assessment, lookup, or simulate. The executor runs the plan against the graph. The LLM never touches the data. It can't invent a shelter address because it never sees the addresses."

**"What's the most impressive technical thing here?"**
> "The PLUTO overflow site discovery. We're using NYC's entire tax lot database — 857,000 records — to find buildings with assembly zoning that could serve as emergency shelters. That's not in any DHS database. We derived it from land use codes, sorted by distance, in 3 seconds. That's cuDF doing what pandas can't."

**"Is this production-ready?"**
> "The architecture is. The data is real — all from NYC Open Data. What's missing for production: live DHS capacity feeds, MTA real-time delays, and a caseworker authentication layer. The intelligence layer is working. The integrations are a sprint, not a research problem."

---

## Timing guide

| Segment | Time | Cumulative |
|---------|------|------------|
| Opening pitch | 0:30 | 0:30 |
| Demo 1 — Tina | 2:30 | 3:00 |
| Demo 2 — Cold emergency | 2:30 | 5:30 |
| Demo 3 — Resource gap | 1:00 | 6:30 |
| Demo 4 — Migrant allocation | 1:00 | 7:30 |
| Technical close | 1:00 | 8:30 |
| Q&A buffer | 1:30 | 10:00 |

**If short on time:** Skip Demo 3 and 4. Tina + Cold Emergency is the full story.
**If running long:** Cut the technical close. Judges can read the Devpost.
