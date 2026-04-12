# nyc-caseworker — OpenClaw Skill

An OpenClaw skill that powers AI agents with the ability to coordinate NYC social services for people in crisis.

## Quick Start

```bash
# Install the parent project's dependencies
cd ../..
pip install -r requirements.txt

# Test the skill
python skills/nyc-caseworker/skill.py
```

## Skill Manifest

See `SKILL.md` for the full YAML frontmatter and skill description.

## Tools Exposed

| Tool | Purpose |
|------|---------|
| `find_resources(query, location, case_id)` | Find NYC resources matching a query, sorted by distance |
| `get_directions(from, to, budget)` | Multi-modal directions (walk/transit) with budget awareness |
| `calculate_eligibility(household)` | SNAP/Medicaid/WIC/Cash Assistance eligibility + estimates |
| `get_rights(resource_type)` | Legal rights at a resource (right to shelter, no ID needed, etc.) |
| `get_stories(need)` | Anonymized success stories for trust-building |
| `case_login(client_id)` | Resume a client's case with contextual summary |
| `case_choose(client_id, need, resource)` | Record client's resource selection |
| `case_checkin(client_id, arrived, resource)` | Confirm arrival; mark as resolved or find alternatives |
| `caseworker_agent(query, ...)` | Full workflow: resources + rights + stories + eligibility |

## Example Agent Integration

```python
from skills.nyc_caseworker.skill import caseworker_agent

# One call, full caseworker response
response = caseworker_agent(
    query="I'm homeless with 2 kids, at 43rd street Brooklyn",
    case_id="client-2024-abc",
    location={"lat": 40.65, "lon": -73.95},
    budget=5.00,
)

print(response["answer"])
for r in response["resources"]:
    print(f"  - {r['name']} ({r['distance_miles']} mi)")
for right in response["rights"]:
    print(f"  ✓ {right['right']}: {right['detail']}")
```

## Data Requirements

This skill expects the parent project's data files:
- `../../data/resource_mart.parquet` (7,759 NYC resources)
- `../../data/graph.pkl` (cuGraph knowledge graph)
- `../../data/cases/` (case files, created on demand)

All data sourced from [NYC Open Data](https://data.cityofnewyork.us/).

## License

MIT
