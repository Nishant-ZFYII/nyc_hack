"""Reproduce /api/admin/cases logic with verbose error reporting."""
import sys, traceback
sys.path.insert(0, ".")

from pipeline.cases import list_cases, load_case
from pipeline.briefing import _estimate_urgency

rows = list_cases()
print(f"list_cases() returned {len(rows)} rows")

for c_summary in rows:
    try:
        case = load_case(c_summary["case_id"])
        if not case:
            print(f"  {c_summary['case_id']}: load_case returned None")
            continue
        needs = case.get("needs", [])
        failed = []
        for n in needs:
            failed.extend(n.get("failed_resources", []))
        urgency = _estimate_urgency(needs, failed)
        print(f"  OK  {c_summary['case_id']:14s} urgency={urgency} needs={len(needs)}")
    except Exception as e:
        print(f"  ERR {c_summary['case_id']}: {type(e).__name__}: {e}")
        traceback.print_exc()
