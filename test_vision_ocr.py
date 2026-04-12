"""
test_vision_ocr.py — Verify llama3.2-vision can read an ID image via Ollama.

Run:
    python3 test_vision_ocr.py [path/to/image.jpg]

Default: samples/sample_id.jpg

Prints:
  - Raw model response
  - Parsed JSON (if the model returns valid JSON)
  - Timing

No changes to the pipeline — this is pure exploration.
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
VISION_MODEL = "llama3.2-vision:11b"

PROMPT = """You are an OCR assistant. Read this US driver's license / state ID image carefully.

Extract the following fields and return ONLY a JSON object (no prose, no markdown):

{
  "first_name": "",
  "last_name": "",
  "full_name": "",
  "dob": "MM/DD/YYYY",
  "address": "",
  "city": "",
  "state": "",
  "zip": "",
  "sex": "M or F",
  "id_number": "",
  "expiration": "MM/DD/YYYY",
  "eye_color": "",
  "hair_color": "",
  "height": "",
  "weight": ""
}

If a field is not visible or unclear, use an empty string "". Do not invent data.
Return ONLY the JSON. No explanation.
"""


def extract_with_vision(image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    t0 = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": VISION_MODEL,
            "messages": [
                {"role": "user", "content": PROMPT, "images": [b64]},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=180,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return {"elapsed_s": elapsed, "raw": content, "usage": data}


def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples/sample_id.jpg")
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)
    print(f"Using image: {img_path}")
    print(f"Model: {VISION_MODEL}")
    print(f"Endpoint: {OLLAMA_URL}")
    print()

    result = extract_with_vision(img_path)
    print(f"--- Inference took {result['elapsed_s']:.1f}s ---\n")

    print("RAW RESPONSE:")
    print(result["raw"])
    print()

    # Try to parse as JSON
    try:
        parsed = json.loads(result["raw"])
        print("PARSED JSON:")
        for k, v in parsed.items():
            print(f"  {k:15s} = {v!r}")
    except json.JSONDecodeError as e:
        print(f"(Not valid JSON: {e})")


if __name__ == "__main__":
    main()
