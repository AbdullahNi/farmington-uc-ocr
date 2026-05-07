import os
import re
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
CORS(app, origins="*")

VISION_API_KEY = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY")
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def call_vision_api(image_bytes):
    if not VISION_API_KEY:
        raise ValueError("GOOGLE_CLOUD_VISION_API_KEY not set")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": encoded},
            "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
        }]
    }

    resp = requests.post(
        VISION_API_URL,
        params={"key": VISION_API_KEY},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    responses = data.get("responses", [])
    if not responses:
        return ""

    full_text = responses[0].get("fullTextAnnotation")
    if full_text:
        return full_text.get("text", "")

    annotations = responses[0].get("textAnnotations", [])
    if annotations:
        return annotations[0].get("description", "")

    return ""


def parse_id_fields(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result = {"full_name": None, "date_of_birth": None, "address": None}

    dob_pattern = re.compile(
        r"(?:dob|date\s+of\s+birth|birth\s*date)[:\s]*"
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        re.IGNORECASE,
    )
    bare_date = re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b")
    zip_pattern = re.compile(r"\b\d{5}(?:-\d{4})?\b")
    address_keywords = re.compile(
        r"\b(?:ave|st|rd|dr|ln|blvd|ct|pl|hwy|way|street|avenue|road|drive)\b",
        re.IGNORECASE,
    )
    all_caps_name = re.compile(r"^[A-Z]{2,}(?:\s+[A-Z]{2,}){1,4}$")

    for line in lines:
        if result["date_of_birth"] is None:
            m = dob_pattern.search(line)
            if m:
                result["date_of_birth"] = m.group(1)

        if result["address"] is None and (
            address_keywords.search(line) or zip_pattern.search(line)
        ):
            result["address"] = line

    if result["date_of_birth"] is None:
        for line in lines:
            m = bare_date.search(line)
            if m:
                result["date_of_birth"] = m.group(1)
                break

    for line in lines[:10]:
        if all_caps_name.match(line):
            result["full_name"] = line.title()
            break

    if result["address"] is None:
        for i, line in enumerate(lines):
            if zip_pattern.search(line):
                parts = [line]
                if i > 0:
                    parts = [lines[i - 1]] + parts
                result["address"] = " ".join(parts)
                break

    return result


def parse_insurance_fields(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result = {"insurance_provider": None, "member_id": None}

    member_pattern = re.compile(
        r"(?:member\s*(?:id|#|no|number)|subscriber\s*id|policy\s*(?:id|no|number))"
        r"[:\s]*([A-Z0-9\-]{4,20})",
        re.IGNORECASE,
    )
    known_providers = re.compile(
        r"\b(aetna|cigna|humana|anthem|blue\s*cross|blue\s*shield|bcbs|kaiser|"
        r"unitedhealthcare|united\s*health|uhc|medicare|medicaid|molina|"
        r"ambetter|priority\s*health|meridian|mclaren|hap|bcbsm)\b",
        re.IGNORECASE,
    )
    fallback_id = re.compile(r"\b([A-Z]{1,3}\d{6,12})\b")

    for line in lines:
        if result["member_id"] is None:
            m = member_pattern.search(line)
            if m:
                result["member_id"] = m.group(1).strip()

        if result["insurance_provider"] is None:
            m = known_providers.search(line)
            if m:
                result["insurance_provider"] = m.group(1).strip().title()

    if result["member_id"] is None:
        for line in lines:
            m = fallback_id.search(line)
            if m:
                result["member_id"] = m.group(1)
                break

    if result["insurance_provider"] is None and lines:
        result["insurance_provider"] = lines[0].strip()

    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/extract-id", methods=["POST"])
def extract_id():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file"}), 400

    try:
        text = call_vision_api(image_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not text:
        return jsonify({"error": "No text detected", "success": False}), 422

    fields = parse_id_fields(text)
    return jsonify({
        "success": True,
        "full_name": fields["full_name"],
        "date_of_birth": fields["date_of_birth"],
        "address": fields["address"],
        "raw_text": text,
    })


@app.route("/extract-insurance", methods=["POST"])
def extract_insurance():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file"}), 400

    try:
        text = call_vision_api(image_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not text:
        return jsonify({"error": "No text detected", "success": False}), 422

    fields = parse_insurance_fields(text)
    return jsonify({
        "success": True,
        "insurance_provider": fields["insurance_provider"],
        "member_id": fields["member_id"],
        "raw_text": text,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
