import os
import re
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)

VISION_API_KEY = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY")
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def call_vision_api(image_bytes: bytes) -> str:
    if not VISION_API_KEY:
        raise ValueError("GOOGLE_CLOUD_VISION_API_KEY environment variable is not set")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": encoded},
                "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
            }
        ]
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

    full_text_annotation = responses[0].get("fullTextAnnotation")
    if full_text_annotation:
        return full_text_annotation.get("text", "")

    text_annotations = responses[0].get("textAnnotations", [])
    if text_annotations:
        return text_annotations[0].get("description", "")

    return ""


def parse_id_fields(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result = {"full_name": None, "date_of_birth": None, "address": None}

    dob_pattern = re.compile(
        r"(?:dob|date\s+of\s+birth|birth\s*date|born)[:\s]*"
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
        re.IGNORECASE,
    )
    bare_date_pattern = re.compile(
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b"
    )

    name_pattern = re.compile(
        r"(?:name|full\s+name|last\s*,?\s*first)[:\s]+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,4})",
        re.IGNORECASE,
    )

    address_keywords = re.compile(
        r"\b(?:address|addr|residence|street|st\.|ave\.|blvd\.|rd\.|dr\.|ln\.|ct\.|pl\.|hwy\.?)\b",
        re.IGNORECASE,
    )
    zip_pattern = re.compile(r"\b\d{5}(?:-\d{4})?\b")

    for line in lines:
        if result["date_of_birth"] is None:
            m = dob_pattern.search(line)
            if m:
                result["date_of_birth"] = m.group(1)

        if result["full_name"] is None:
            m = name_pattern.search(line)
            if m:
                result["full_name"] = m.group(1).strip()

        if result["address"] is None and (
            address_keywords.search(line) or zip_pattern.search(line)
        ):
            result["address"] = line

    if result["date_of_birth"] is None:
        for line in lines:
            m = bare_date_pattern.search(line)
            if m:
                result["date_of_birth"] = m.group(1)
                break

    if result["full_name"] is None:
        all_caps_name = re.compile(r"^[A-Z]{2,}(?:\s+[A-Z]{2,}){1,4}$")
        for line in lines[:10]:
            if all_caps_name.match(line):
                result["full_name"] = line.title()
                break

    if result["address"] is None:
        for i, line in enumerate(lines):
            if zip_pattern.search(line):
                addr_parts = [line]
                if i > 0 and re.search(r"\d+\s+\w+", lines[i - 1]):
                    addr_parts = [lines[i - 1]] + addr_parts
                result["address"] = " ".join(addr_parts)
                break

    return result


def parse_insurance_fields(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result = {"insurance_provider": None, "member_id": None}

    member_id_pattern = re.compile(
        r"(?:member\s*(?:id|#|no\.?|number)|id\s*(?:#|no\.?|number)|subscriber\s*id|policy\s*(?:id|no\.?|number)|group\s*(?:id|no\.?|number))[:\s]*([A-Z0-9\-]{4,20})",
        re.IGNORECASE,
    )

    known_providers = re.compile(
        r"\b(aetna|cigna|humana|anthem|blue\s*cross|blue\s*shield|bcbs|kaiser|unitedhealthcare|united\s*health|uhc|medicare|medicaid|molina|centene|cvs\s*health|oscar|ambetter|bright\s*health|devoted\s*health|health\s*net|magellan|wellcare)\b",
        re.IGNORECASE,
    )

    for line in lines:
        if result["member_id"] is None:
            m = member_id_pattern.search(line)
            if m:
                result["member_id"] = m.group(1).strip()

        if result["insurance_provider"] is None:
            m = known_providers.search(line)
            if m:
                result["insurance_provider"] = m.group(1).strip().title()

    if result["insurance_provider"] is None:
        provider_label = re.compile(
            r"(?:insurance|insurer|carrier|plan|company|provider)[:\s]+([A-Za-z][A-Za-z0-9\s&,\.'\-]{2,50})",
            re.IGNORECASE,
        )
        for line in lines:
            m = provider_label.search(line)
            if m:
                candidate = m.group(1).strip().rstrip(".,")
                if 2 < len(candidate) < 60:
                    result["insurance_provider"] = candidate
                    break

    if result["member_id"] is None:
        fallback_id = re.compile(r"\b([A-Z]{1,3}\d{6,12})\b")
        for line in lines:
            m = fallback_id.search(line)
            if m:
                result["member_id"] = m.group(1)
                break

    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/extract-id", methods=["POST"])
def extract_id():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send the file as 'image' in a multipart/form-data request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file."}), 400

    try:
        text = call_vision_api(image_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except requests.HTTPError as e:
        return jsonify({"error": f"Vision API error: {e.response.status_code} {e.response.text}"}), 502
    except Exception as e:
        return jsonify({"error": f"Failed to call Vision API: {str(e)}"}), 502

    if not text:
        return jsonify({"error": "No text detected in the image."}), 422

    fields = parse_id_fields(text)
    return jsonify({
        "full_name": fields["full_name"],
        "date_of_birth": fields["date_of_birth"],
        "address": fields["address"],
        "raw_text": text,
    })


@app.route("/extract-insurance", methods=["POST"])
def extract_insurance():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send the file as 'image' in a multipart/form-data request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file."}), 400

    try:
        text = call_vision_api(image_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except requests.HTTPError as e:
        return jsonify({"error": f"Vision API error: {e.response.status_code} {e.response.text}"}), 502
    except Exception as e:
        return jsonify({"error": f"Failed to call Vision API: {str(e)}"}), 502

    if not text:
        return jsonify({"error": "No text detected in the image."}), 422

    fields = parse_insurance_fields(text)
    return jsonify({
        "insurance_provider": fields["insurance_provider"],
        "member_id": fields["member_id"],
        "raw_text": text,
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
