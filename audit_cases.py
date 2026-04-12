"""Quick audit of existing cases — shows what's in data/cases/."""
from pipeline.cases import list_cases, load_case

rows = list_cases()
print(f"Total cases: {len(rows)}\n")

for cs in rows:
    c = load_case(cs["case_id"])
    if not c:
        continue
    needs = c.get("needs", [])
    cats = [n.get("category") for n in needs]
    open_needs = [n.get("category") for n in needs if n.get("status") != "resolved"]
    tickets = c.get("tickets", [])
    visits = c.get("visits", [])
    dests = c.get("destination_intents", [])
    name = (c.get("name") or "?")[:18]
    case_id = c.get("case_id", "?")[:28]
    print(f"{case_id:30s} {name:20s} "
          f"needs={cats[:3] if cats else '[]':<30s} "
          f"open={len(open_needs)}/{len(cats)} "
          f"visits={len(visits)} "
          f"tickets={len(tickets)} "
          f"dests={len(dests)}")
