"""
admin_server.py — Standalone FastAPI admin portal for NYC Social Services Intelligence Engine.

Runs on a separate port from the user-facing server.py.
Shares all pipeline/, engine/, llm/, and data/ modules with server.py.
The data link is data/cases/*.json — written by server.py, read here.

Run:
    uvicorn admin_server:app --host 0.0.0.0 --port 9001

User portal (server.py) runs on port 9000.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.cases import (
    load_case,
    list_cases,
    update_destination_state,
    update_need_status,
    save_admin_notes,
)
from pipeline.briefing import generate_briefing, _estimate_urgency
from pipeline.executor import load_state
from pipeline.form_filler import fill_forms_from_id, extract_id_fields

app = FastAPI(title="NYC Social Services — Admin Portal")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(ROOT / "frontend" / "admin.html")


# ── Request models ─────────────────────────────────────────

class BriefingRequest(BaseModel):
    case_id: str
    resource_name: str = ""
    resource_type: str = ""
    resource_address: str = ""


class AdminStateRequest(BaseModel):
    case_id: str
    resource_name: str
    new_state: str


class AdminNotesRequest(BaseModel):
    case_id: str
    notes: str


class AdminNeedStatusRequest(BaseModel):
    case_id: str
    category: str
    status: str


# ── Admin endpoints ────────────────────────────────────────

@app.post("/api/admin/briefing")
async def admin_briefing(req: BriefingRequest):
    """Generate AI-powered admin briefing for an incoming client.

    Called when a user confirms destination intent. Returns a structured
    briefing with diagnosis, urgency, document requirements with gap detection,
    edge cases with workarounds, and recommended handling approach.
    """
    case = load_case(req.case_id)
    if not case:
        raise HTTPException(404, f"Case '{req.case_id}' not found")

    resource = {
        "name": req.resource_name,
        "resource_type": req.resource_type,
        "address": req.resource_address,
    }

    # Enrich resource details from mart if resource_name provided
    if req.resource_name:
        try:
            import pandas as pd
            mart, _ = load_state()
            mart_pd = mart.to_pandas() if hasattr(mart, "to_pandas") else mart
            match = mart_pd[mart_pd["name"].str.contains(req.resource_name, case=False, na=False)]
            if len(match):
                row = match.iloc[0]
                if not resource["resource_type"]:
                    resource["resource_type"] = str(row.get("resource_type", ""))
                if not resource["address"]:
                    resource["address"] = str(row.get("address", ""))
                resource["resource_id"] = str(row.get("resource_id", ""))
        except Exception:
            pass

    return generate_briefing(case, resource)


@app.get("/api/admin/cases")
async def admin_cases():
    """Enhanced case list for admin dashboard.

    Returns all cases with urgency classification, active destination intents,
    and emergency contact status — sorted by urgency then last visit.
    """
    urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    cases_raw = list_cases()
    enhanced = []

    for c_summary in cases_raw:
        case = load_case(c_summary["case_id"])
        if not case:
            continue

        needs = case.get("needs", [])
        failed: list = []
        for n in needs:
            failed.extend(n.get("failed_resources", []))

        active_dests = [
            i for i in case.get("destination_intents", [])
            if i.get("state") not in {"resolved", "cancelled"}
        ]

        ec = case.get("emergency_contact", {})
        ec_registered = bool(isinstance(ec, dict) and ec.get("telegram_chat_id"))

        urgency = _estimate_urgency(needs, failed)

        enhanced.append({
            **c_summary,
            "urgency": urgency,
            "urgency_order": urgency_order.get(urgency, 3),
            "active_destinations": active_dests,
            "active_destination_count": len(active_dests),
            "failed_resource_count": len(failed),
            "emergency_contact": ec if isinstance(ec, dict) else {},
            "ec_registered": ec_registered,
            "current_location": case.get("current_location"),
            "open_needs_list": [
                {"category": n.get("category"), "priority": n.get("priority", 99)}
                for n in sorted(needs, key=lambda x: x.get("priority", 99))
                if n.get("status") != "resolved"
            ],
        })

    enhanced.sort(key=lambda x: (x["urgency_order"], -(len(x.get("last_visit", "")))))
    return enhanced


@app.get("/api/admin/case/{case_id}")
async def admin_case_detail(case_id: str):
    """Full case record for admin briefing panel."""
    case = load_case(case_id)
    if not case:
        raise HTTPException(404, f"Case '{case_id}' not found")
    return case


@app.post("/api/admin/update_state")
async def admin_update_state(req: AdminStateRequest):
    """Admin manually advances destination lifecycle state."""
    update_destination_state(req.case_id, req.resource_name, req.new_state)
    return {
        "ok": True,
        "case_id": req.case_id,
        "resource_name": req.resource_name,
        "new_state": req.new_state,
    }


@app.post("/api/admin/notes")
async def admin_save_notes(req: AdminNotesRequest):
    """Save admin notes to a case."""
    result = save_admin_notes(req.case_id, req.notes)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return {"ok": True, "case_id": req.case_id}


@app.post("/api/admin/need_status")
async def admin_need_status(req: AdminNeedStatusRequest):
    """Admin updates a need's status (open / in_progress / resolved)."""
    result = update_need_status(req.case_id, req.category, req.status)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return {"ok": True, "case_id": req.case_id, "category": req.category, "status": req.status}


@app.get("/api/admin/stats")
async def admin_stats():
    """Aggregate statistics across all cases for the admin dashboard."""
    cases_raw = list_cases()
    urgency_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    need_category_counts: dict[str, int] = {}
    total_visits = 0
    resolved_needs = 0
    open_needs = 0

    for c_summary in cases_raw:
        case = load_case(c_summary["case_id"])
        if not case:
            continue
        total_visits += len(case.get("visits", []))
        for n in case.get("needs", []):
            cat = n.get("category", "other")
            need_category_counts[cat] = need_category_counts.get(cat, 0) + 1
            if n.get("status") == "resolved":
                resolved_needs += 1
            else:
                open_needs += 1

    # Urgency counts come from the enhanced cases endpoint logic
    for c_summary in cases_raw:
        case = load_case(c_summary["case_id"])
        if not case:
            continue
        needs = case.get("needs", [])
        failed: list = []
        for n in needs:
            failed.extend(n.get("failed_resources", []))
        urg = _estimate_urgency(needs, failed)
        urgency_counts[urg] = urgency_counts.get(urg, 0) + 1

    top_needs = sorted(need_category_counts.items(), key=lambda x: -x[1])[:8]

    return {
        "total_cases": len(cases_raw),
        "total_visits": total_visits,
        "open_needs": open_needs,
        "resolved_needs": resolved_needs,
        "urgency_counts": urgency_counts,
        "top_need_categories": [{"category": k, "count": v} for k, v in top_needs],
    }


# ── Form Filler — OCR + pre-filled PDFs ───────────────────────────────────────

@app.post("/api/admin/fill_forms")
async def fill_forms(case_id: str = Form(...),
                     forms: str = Form("snap,medicaid,proof"),
                     id_image: UploadFile = File(...)):
    """
    Upload an ID photo + case_id. Returns a ZIP of pre-filled PDFs.

    The flow:
      1. OCR the uploaded ID (tesseract)
      2. Load case data (household size, income)
      3. Generate requested forms as PDFs
      4. Return ZIP
    """
    import zipfile, tempfile, os

    # Save uploaded image
    tmp_dir = tempfile.mkdtemp()
    id_path = os.path.join(tmp_dir, id_image.filename or "id.jpg")
    with open(id_path, "wb") as f:
        f.write(await id_image.read())

    # Load case data for household + income
    case = load_case(case_id) or {}
    visits = case.get("visits", [])
    # Try to extract profile from latest visit's plan
    profile = {}
    if visits:
        last_plan = visits[-1].get("plan", {}) or {}
        profile = last_plan.get("client_profile", {}) or {}

    case_data = {
        "household_size": profile.get("household_size", 1),
        "annual_income": profile.get("income", 0) or profile.get("annual_income", 0),
        "has_children": profile.get("has_children", False),
        "housing_status": profile.get("housing_status", ""),
        "snap_estimate": 598,  # demo placeholder; real eligibility calc happens in form
    }

    form_types = [f.strip() for f in forms.split(",") if f.strip()]
    result = fill_forms_from_id(id_path, case_data=case_data, forms=form_types)

    # Bundle PDFs into a ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for form_type, pdf_bytes in result["forms"].items():
            zf.writestr(f"{form_type}_{case_id}.pdf", pdf_bytes)
        # Also include extracted ID fields as JSON
        import json as _json
        zf.writestr("extracted_id_fields.json",
                    _json.dumps(result["id_fields"], indent=2))

    zip_buf.seek(0)
    return StreamingResponse(
        iter([zip_buf.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="forms_{case_id}.zip"'},
    )


@app.post("/api/admin/ocr_id")
async def ocr_id(id_image: UploadFile = File(...)):
    """
    OCR an ID image without generating forms.
    Returns extracted fields as JSON. Useful for preview before form-fill.
    """
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(await id_image.read())
    tmp.close()
    try:
        fields = extract_id_fields(tmp.name)
    finally:
        os.unlink(tmp.name)
    return fields
