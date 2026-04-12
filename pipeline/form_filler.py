"""
pipeline/form_filler.py — Extract ID data + pre-fill NYC benefits forms.

Uses tesseract OCR on an ID image to extract:
  - name (first, last)
  - date of birth
  - address
  - ID/license number
  - expiration
  - sex, eye color, height, weight

Then generates a pre-filled PDF for:
  - SNAP application (NY LDSS-4826)
  - Medicaid application (DOH-4220)
  - HRA Request for Proof (for users without full docs)

Uses reportlab to build the PDF directly (no form template needed).
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image


# ── ID field extraction ──────────────────────────────────────────────────────

def extract_id_fields(image_path: str | Path) -> dict:
    """Run OCR on an ID image and extract structured fields.

    Parser is tuned for standard US driver's license layout. Uses line-aware
    parsing + strict patterns to avoid false matches like "LICENSE" → city.
    """
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    # Words we must NEVER confuse for first/last names or cities
    _DENY = {"LICENSE", "DRIVER", "LICENCE", "CLASS", "STATE", "CALIFORNIA",
             "NEW", "YORK", "CARD", "IDENTIFICATION", "DONOR", "VETERAN",
             "FEDERAL", "LIMITS", "APPLY", "VEH", "REAL", "ID"}

    fields = {
        "raw_ocr": raw_text,
        "first_name": "",
        "last_name": "",
        "full_name": "",
        "dob": "",
        "address": "",
        "city": "",
        "state": "",
        "zip": "",
        "id_number": "",
        "expiration": "",
        "sex": "",
        "eye_color": "",
        "hair_color": "",
        "height": "",
        "weight": "",
    }

    def _clean_token(s: str) -> str:
        s = s.strip().rstrip(',').rstrip('.')
        # Drop anything in deny list or with digits
        if not s or s.upper() in _DENY:
            return ""
        if any(c.isdigit() for c in s):
            return ""
        return s.upper()

    # DL/ID number — tesseract commonly misreads "DL" as "pt", "DI", "D|"
    # Accept common prefixes; the value is 7-12 digits
    m = re.search(r'(?:\bDL\b|\bpt\b|\bDI\b|\bD\|)\s*[:#]?\s*(\d{7,12})\b', raw_text, re.IGNORECASE)
    if m:
        fields["id_number"] = m.group(1)
    else:
        for candidate in re.findall(r'\b(\d{7,12})\b', raw_text):
            if len(candidate) in (7, 8, 9):
                fields["id_number"] = candidate
                break

    # DOB — tesseract misreads "DOB" as "p08", "008", "DOP", "pos"
    m = re.search(r'(?:\bDOB\b|\bp08\b|\b008\b|\bDOP\b|\bpos\b)\s*[:]?\s*(\d{2}/\d{2}/\d{4})\b', raw_text, re.IGNORECASE)
    if m:
        fields["dob"] = m.group(1)

    # Expiration — "EXP" misread as "exe", "EXe", "£XP"
    m = re.search(r'(?:\bEXP\b|\bexe\b|\b£XP\b)\s*[:]?\s*(\d{2}/\d{2}/\d{4})\b', raw_text, re.IGNORECASE)
    if m:
        fields["expiration"] = m.group(1)

    # Last name — "LN" is commonly misread as "in", "1n", "LN"
    # The label + name may have NO space between them: "inDOE"
    # Match: (ln|in|1n)[space or nothing][UPPERCASE WORD]
    m = re.search(r'(?:\bLN\b|\bin\b|\b1n\b|(?<=\n)in|(?<=^)in)\s*([A-Z][A-Z\-\' ]{1,40})', raw_text)
    if m:
        candidate = _clean_token(m.group(1).split('\n')[0])
        if candidate:
            fields["last_name"] = candidate
    else:
        # Fallback: tight pattern "inWORD" (no space, common tesseract issue)
        m = re.search(r'\b[il1]n([A-Z]{2,})\b', raw_text)
        if m:
            candidate = _clean_token(m.group(1))
            if candidate:
                fields["last_name"] = candidate

    # First name — "FN" misread as "rn", "Fn", "FR", "FIN"
    # Similar no-space issue: "rnJOHN"
    m = re.search(r'(?:\bFN\b|\brn\b|\bFR\b|\bFIN\b)\s*([A-Z][A-Z\-\' ]{1,30})', raw_text)
    if m:
        candidate = _clean_token(m.group(1).split('\n')[0])
        if candidate and candidate != fields["last_name"]:
            fields["first_name"] = candidate
    else:
        # Fallback: tight "rnWORD" pattern
        m = re.search(r'\brn([A-Z]{2,})\b', raw_text)
        if m:
            candidate = _clean_token(m.group(1))
            if candidate and candidate != fields["last_name"]:
                fields["first_name"] = candidate

    # Address — line starting with 2-5 digits + street words + common suffix
    # or a number line followed by words
    addr_pattern = re.compile(
        r'\b(\d{1,5}\s+[A-Z][A-Z0-9 \-\.\']{3,60}?(?:STREET|ST|AVENUE|AVE|BLVD|BOULEVARD|ROAD|RD|LANE|LN|DRIVE|DR|WAY|COURT|CT|PLACE|PL|PARKWAY|PKWY|TERRACE|TER)\b)',
        re.IGNORECASE,
    )
    m = addr_pattern.search(raw_text)
    if m:
        fields["address"] = m.group(1).strip().upper()
    else:
        # Fallback: a line that starts with digits followed by all-caps words
        for ln in lines:
            m2 = re.match(r'^(\d{2,5}\s+[A-Z][A-Z\- ]{3,50})$', ln)
            if m2:
                fields["address"] = m2.group(1).strip().upper()
                break

    # City, State ZIP — "CITY, ST 12345" OR "CITY, ST012345" (no space, common on DLs)
    # Accept 5-9 digit zip to handle OCR smushing state+zip together
    m = re.search(r'([A-Z][A-Z \-]{2,30}),\s+([A-Z]{2})\s*(\d{5,9})', raw_text)
    if m:
        city = _clean_token(m.group(1))
        state = m.group(2).upper()
        # Reject obvious non-states (DL ID prefixes like "PT")
        valid_us_states = {
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
            "DC",
        }
        if city and state in valid_us_states:
            fields["city"] = city
            fields["state"] = state
            fields["zip"] = m.group(3)

    # Sex
    m = re.search(r'\bSEX\b\s*[:]?\s*([MF])\b', raw_text, re.IGNORECASE)
    if m:
        fields["sex"] = m.group(1).upper()

    # Eye color
    m = re.search(r'\bEYES?\b\s*[:]?\s*([A-Z]{3,5})', raw_text, re.IGNORECASE)
    if m:
        fields["eye_color"] = m.group(1).strip().upper()

    # Hair color
    m = re.search(r'\bHAIR\b\s*[:]?\s*([A-Z]{3,5})', raw_text, re.IGNORECASE)
    if m:
        fields["hair_color"] = m.group(1).strip().upper()

    # Height — 6'0"
    m = re.search(r"\bHGT\b\s*[:]?\s*(\d'[\s]?\d{1,2}\"?)", raw_text, re.IGNORECASE)
    if m:
        fields["height"] = m.group(1).strip()

    # Weight
    m = re.search(r"\bWGT\b\s*[:]?\s*(\d{2,3})\s*lb", raw_text, re.IGNORECASE)
    if m:
        fields["weight"] = f"{m.group(1)} lb"

    # Full name
    parts = [fields["first_name"], fields["last_name"]]
    fields["full_name"] = " ".join(p for p in parts if p).strip()

    return fields


# ── Real-form overlay using official NYC HRA / NYS DOH PDFs ────────────────────
# These are the actual government forms a caseworker would hand a client.
# We find label positions with pdfplumber, then overlay filled values using
# reportlab and merge with pypdf — producing a "filled" version of the real form.

FORMS_DIR = Path(__file__).resolve().parent.parent / "samples" / "forms"
SNAP_PDF = FORMS_DIR / "ldss_4826_snap.pdf"
MEDICAID_PDF = FORMS_DIR / "doh_4220_medicaid.pdf"


def _overlay_on_real_form(source_pdf: Path, answers: dict) -> bytes:
    """
    Overlay filled answers onto the source PDF.

    `answers` is a dict of {label_substring: value}. For each label match
    found via pdfplumber text extraction, the value is drawn a bit to the
    right of the label's bounding box.

    Returns the merged PDF as bytes.
    """
    import pdfplumber
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas as rl_canvas
    from io import BytesIO

    # Pass 1: find (page_idx, x, y) placement points for each answer
    placements = []  # list of (page_idx, x, y, text)
    used_labels = set()  # don't place a value multiple times per label

    with pdfplumber.open(source_pdf) as pdf:
        n_pages = len(pdf.pages)
        page_sizes = [(p.width, p.height) for p in pdf.pages]

        for page_idx, page in enumerate(pdf.pages):
            words = page.extract_words()
            # Build lower-cased text index for fuzzy label matching
            for label_key, value in answers.items():
                if label_key in used_labels or not value:
                    continue
                label_tokens = label_key.lower().split()
                # Find sequential tokens on the page
                for i, w in enumerate(words):
                    if w["text"].lower() == label_tokens[0]:
                        # Check following tokens match
                        ok = True
                        last_w = w
                        for j, t in enumerate(label_tokens[1:], 1):
                            if i + j >= len(words) or words[i + j]["text"].lower() != t:
                                ok = False
                                break
                            last_w = words[i + j]
                        if ok:
                            # Place text just right of the last matched token
                            # pdfplumber "top" is from top; reportlab y from bottom
                            x = last_w["x1"] + 6
                            y = page_sizes[page_idx][1] - last_w["top"] - 10
                            placements.append((page_idx, x, y, str(value)))
                            used_labels.add(label_key)
                            break

    # Pass 2: build overlay PDF with reportlab (one page per source page)
    overlay_buf = BytesIO()
    c = rl_canvas.Canvas(overlay_buf)
    for page_idx in range(n_pages):
        w, h = page_sizes[page_idx]
        c.setPageSize((w, h))
        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(0.05, 0.35, 0.05)  # dark green to distinguish filled text
        for pi, x, y, txt in placements:
            if pi == page_idx:
                c.drawString(x, y, txt)
        c.showPage()
    c.save()
    overlay_buf.seek(0)

    # Pass 3: merge overlay onto original
    base = PdfReader(str(source_pdf))
    overlay = PdfReader(overlay_buf)
    writer = PdfWriter()
    for i, page in enumerate(base.pages):
        if i < len(overlay.pages):
            page.merge_page(overlay.pages[i])
        writer.add_page(page)

    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _answers_from_case(id_fields: dict, case_data: dict) -> dict:
    """Map extracted ID fields + case data to form-label-friendly answers."""
    fn = id_fields.get("first_name", "") or ""
    ln = id_fields.get("last_name", "") or ""
    full = id_fields.get("full_name", "") or f"{fn} {ln}".strip()
    addr = id_fields.get("address", "")
    city = id_fields.get("city", "")
    state = id_fields.get("state", "")
    zipc = id_fields.get("zip", "")
    dob = id_fields.get("dob", "")
    sex = id_fields.get("sex", "")
    household = case_data.get("household_size", "")
    income = case_data.get("annual_income", "")

    # Keys use the label text as it appears in the form (lowercased). The
    # overlay function does case-insensitive token matching.
    return {
        "last name": ln,
        "first name": fn,
        "date of birth": dob,
        "sex": sex,
        "address": addr,
        "city": city,
        "state": state,
        "zip": zipc,
        "zip code": zipc,
        "home address": addr,
        "mailing address": addr,
        "home phone": "",
        "social security number": "",
        "household size": str(household) if household else "",
        "annual income": f"${income:,}" if isinstance(income, (int, float)) and income else "",
        "name": full,
        "full name": full,
    }


def generate_snap_form(id_fields: dict, case_data: dict = None) -> bytes:
    """Pre-fill the REAL NYC LDSS-4826 SNAP application PDF."""
    case_data = case_data or {}
    if SNAP_PDF.exists():
        answers = _answers_from_case(id_fields, case_data)
        return _overlay_on_real_form(SNAP_PDF, answers)
    # Fallback to synthetic form if the real PDF isn't available
    return _generate_form(
        title="SNAP APPLICATION (Food Stamps)",
        subtitle="NY LDSS-4826 (Pre-filled)",
        form_type="snap",
        id_fields=id_fields,
        case_data=case_data,
    )


def generate_medicaid_form(id_fields: dict, case_data: dict = None) -> bytes:
    """Pre-fill the REAL NYS DOH-4220 Medicaid application PDF."""
    case_data = case_data or {}
    if MEDICAID_PDF.exists():
        answers = _answers_from_case(id_fields, case_data)
        return _overlay_on_real_form(MEDICAID_PDF, answers)
    return _generate_form(
        title="MEDICAID APPLICATION",
        subtitle="NY DOH-4220 (Pre-filled)",
        form_type="medicaid",
        id_fields=id_fields,
        case_data=case_data,
    )


def generate_request_for_proof(id_fields: dict, case_data: dict = None) -> bytes:
    """HRA Request-for-Proof form — still synthetic (no single canonical PDF)."""
    return _generate_form(
        title="HRA REQUEST FOR PROOF",
        subtitle="Apply for benefits without full documents",
        form_type="proof",
        id_fields=id_fields,
        case_data=case_data or {},
    )


def _generate_form(title: str, subtitle: str, form_type: str,
                   id_fields: dict, case_data: dict) -> bytes:
    """Build a PDF using reportlab with pre-filled fields."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, black, white

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # ── Header ──
    c.setFillColor(HexColor("#76B900"))  # NVIDIA green
    c.rect(0, height - 1.0 * inch, width, 1.0 * inch, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(0.5 * inch, height - 0.55 * inch, title)
    c.setFont("Helvetica", 10)
    c.drawString(0.5 * inch, height - 0.8 * inch, subtitle)
    c.drawRightString(width - 0.5 * inch, height - 0.8 * inch,
                      f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── Personal info section ──
    y = height - 1.5 * inch
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.5 * inch, y, "APPLICANT INFORMATION")
    y -= 0.3 * inch

    c.setFont("Helvetica", 10)
    rows = [
        ("First Name", id_fields.get("first_name", "")),
        ("Last Name", id_fields.get("last_name", "")),
        ("Date of Birth", id_fields.get("dob", "")),
        ("ID / License #", id_fields.get("id_number", "")),
        ("Sex", id_fields.get("sex", "")),
        ("Address", id_fields.get("address", "")),
        ("City", id_fields.get("city", "")),
        ("State", id_fields.get("state", "")),
        ("ZIP Code", id_fields.get("zip", "")),
    ]

    for label, value in rows:
        c.drawString(0.7 * inch, y, f"{label}:")
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2.2 * inch, y, value or "__________________________")
        c.setFont("Helvetica", 10)
        y -= 0.25 * inch

    # ── Case-specific section ──
    y -= 0.2 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.5 * inch, y, "HOUSEHOLD INFORMATION")
    y -= 0.3 * inch

    c.setFont("Helvetica", 10)
    household_rows = [
        ("Household size", str(case_data.get("household_size", "____"))),
        ("Annual income", f"${case_data.get('annual_income', '____'):,}"
            if isinstance(case_data.get("annual_income"), (int, float)) else "$____"),
        ("Has children", "Yes" if case_data.get("has_children") else "No"),
        ("Current housing", case_data.get("housing_status", "____") or "____"),
    ]
    for label, value in household_rows:
        c.drawString(0.7 * inch, y, f"{label}:")
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2.5 * inch, y, str(value))
        c.setFont("Helvetica", 10)
        y -= 0.25 * inch

    # ── Form-specific section ──
    y -= 0.2 * inch
    c.setFont("Helvetica-Bold", 12)

    if form_type == "snap":
        c.drawString(0.5 * inch, y, "SNAP ELIGIBILITY")
        y -= 0.3 * inch
        c.setFont("Helvetica", 9)
        snap_info = [
            f"Estimated monthly benefit: ${case_data.get('snap_estimate', 0)}",
            "Income limit: 130% of Federal Poverty Level",
            "Timeline: 30 days (7 days for emergency SNAP)",
            "Documents needed: ID, proof of income, Social Security numbers",
            "Where to apply: HRA Benefits Access Center",
        ]
    elif form_type == "medicaid":
        c.drawString(0.5 * inch, y, "MEDICAID ELIGIBILITY")
        y -= 0.3 * inch
        c.setFont("Helvetica", 9)
        snap_info = [
            "Free health coverage for qualifying New Yorkers",
            "Income limit: 138% FPL (adults), 223% FPL (pregnant/children)",
            "Timeline: 45 days (24 hours for emergency Medicaid)",
            "Documents: ID, proof of income, proof of NY residence",
            "Where to apply: HRA Medicaid office or NY State of Health",
        ]
    else:  # proof
        c.drawString(0.5 * inch, y, "REQUEST FOR PROOF")
        y -= 0.3 * inch
        c.setFont("Helvetica", 9)
        snap_info = [
            "Use this form to apply for benefits WITHOUT a government ID.",
            "HRA staff will help gather documents on your behalf.",
            "Legal basis: NY Social Services Law §131(a).",
            "Your info cannot be shared with ICE (NYC Executive Order 170).",
            "Where to submit: Any HRA Benefits Access Center.",
        ]

    for line in snap_info:
        c.drawString(0.7 * inch, y, "• " + line)
        y -= 0.22 * inch

    # ── Signature line ──
    y -= 0.3 * inch
    c.setFont("Helvetica", 10)
    c.line(0.7 * inch, y, 4.0 * inch, y)
    c.drawString(0.7 * inch, y - 0.15 * inch, "Applicant signature")
    c.line(4.5 * inch, y, 7.0 * inch, y)
    c.drawString(4.5 * inch, y - 0.15 * inch, "Date")

    # ── Footer ──
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(HexColor("#888888"))
    c.drawString(0.5 * inch, 0.4 * inch,
                 "Pre-filled by NYC Social Services Intelligence Engine "
                 "| Powered by NVIDIA Nemotron + OpenClaw")
    c.drawRightString(width - 0.5 * inch, 0.4 * inch,
                      "Review carefully before submitting")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


# ── One-shot pipeline ────────────────────────────────────────────────────────

def fill_forms_from_id(id_image_path: str | Path, case_data: dict = None,
                       forms: list = None) -> dict:
    """
    Full pipeline: OCR the ID image, then generate requested forms.

    Args:
        id_image_path: Path to the ID image (JPEG/PNG)
        case_data: Optional dict with household_size, annual_income, etc.
        forms: List of form types to generate. Default: all 3.

    Returns:
        {"id_fields": {...}, "forms": {form_type: pdf_bytes, ...}}
    """
    forms = forms or ["snap", "medicaid", "proof"]
    id_fields = extract_id_fields(id_image_path)
    case_data = case_data or {}

    generators = {
        "snap": generate_snap_form,
        "medicaid": generate_medicaid_form,
        "proof": generate_request_for_proof,
    }

    form_pdfs = {}
    for form_type in forms:
        if form_type in generators:
            form_pdfs[form_type] = generators[form_type](id_fields, case_data)

    return {
        "id_fields": id_fields,
        "forms": form_pdfs,
    }


if __name__ == "__main__":
    # Quick test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/sample_id.jpg"
    print(f"Testing form filler on: {path}")

    result = fill_forms_from_id(path, case_data={
        "household_size": 5,
        "annual_income": 28000,
        "has_children": True,
        "housing_status": "at_risk",
        "snap_estimate": 598,
    })

    print("\n=== Extracted ID fields ===")
    for k, v in result["id_fields"].items():
        if k != "raw_ocr" and v:
            print(f"  {k:15s} {v}")

    print(f"\n=== Generated {len(result['forms'])} forms ===")
    for form_type, pdf_bytes in result["forms"].items():
        out_path = f"/tmp/filled_{form_type}.pdf"
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"  {form_type:10s} → {out_path} ({len(pdf_bytes)/1024:.1f} KB)")
