from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import logging
import logging.handlers
import json
import os
import time
import uuid
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)

LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


log = logging.getLogger("sakh-ai")
log.setLevel(logging.INFO)

_console_handler = logging.StreamHandler()
_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
)
_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_console_handler.setFormatter(_formatter)
_file_handler.setFormatter(_formatter)

if not log.handlers:
    log.addHandler(_console_handler)
    log.addHandler(_file_handler)
log.propagate = False


blackbox = logging.getLogger("sakh-ai.blackbox")
blackbox.setLevel(logging.INFO)

_blackbox_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "blackbox.log"),
    maxBytes=20 * 1024 * 1024,  # 20 MB
    backupCount=10,
)
_blackbox_handler.setFormatter(logging.Formatter("%(message)s"))

if not blackbox.handlers:
    blackbox.addHandler(_blackbox_handler)
blackbox.propagate = False


def record_blackbox(**fields) -> None:
    """Write one structured JSON-line record to the blackbox log."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    try:
        blackbox.info(json.dumps(entry, default=str))
    except Exception:
        log.exception("Failed to write blackbox record")


# MODEL_PATH = "Lightgbm_credit2.pkl"
# FEATURES_PATH = "Model_feature2.pkl"

# model = joblib.load(MODEL_PATH)
# model_features = joblib.load(FEATURES_PATH)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "Lightgbm_credit2.pkl")
feature_path = os.path.join(BASE_DIR, "Model_feature2.pkl")

model = joblib.load(model_path)

model_feature_list = joblib.load(feature_path)

EMPLOYMENT_CATEGORIES = ["Commercial associate", "Pensioner", "State servant", "Working"]

def _normalize_key(s: str) -> str:
    return s.strip().lower().replace("-", "").replace("_", "").replace(" ", "")

EMPLOYMENT_TYPE_LOOKUP = {_normalize_key(c): c for c in EMPLOYMENT_CATEGORIES}

APPROVAL_THRESHOLD = 0.35

class ValidationError(Exception):
    pass

def parse_and_validate(payload: dict) -> dict:
    def require(key, cast, lo=None, hi=None, label=None):
        label = label or key
        if key not in payload or payload[key] in (None, ""):
            raise ValidationError(f"Missing field: {label}")
        try:
            value = cast(payload[key])
        except (TypeError, ValueError):
            raise ValidationError(f"Invalid value for {label}")
        if lo is not None and value < lo:
            raise ValidationError(f"{label} must be >= {lo}")
        if hi is not None and value > hi:
            raise ValidationError(f"{label} must be <= {hi}")
        return value

    age = require("age", int, 18, 100, "age")
    city_tier = require("city_tier", int, 1, 3, "city_tier")
    monthly_income = require("salary", float, 0.01, None, "salary")
    cibil_score = require("cibil_score", int, 300, 900, "cibil_score")
    foir_percentage = require("foir_score", float, 0, 100, "foir")

    raw_employment = payload.get("employment_type", "")
    employment_type = EMPLOYMENT_TYPE_LOOKUP.get(_normalize_key(str(raw_employment)))
    if employment_type is None:
        raise ValidationError(
            f"Invalid employment_type: '{raw_employment}'. "
            f"Expected one of: {EMPLOYMENT_CATEGORIES}"
        )

    return {
        "age": age,
        "city_tier": city_tier,
        "employment_type": employment_type,
        "monthly_income": monthly_income,
        "cibil_score": cibil_score,
        "foir_percentage": foir_percentage,
    }

def run_model(input_features: dict):
    df = pd.DataFrame([input_features])
    df["employment_type"] = pd.Categorical(
        df["employment_type"], categories=EMPLOYMENT_CATEGORIES
    )
    df = df[model_feature_list]

    probability_approved = float(model.predict_proba(df)[0][1])
    approved = probability_approved >= APPROVAL_THRESHOLD
    return approved, probability_approved

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/docs")
def docs():
    return render_template("docs.html")

@app.route("/check")
def check():
    return render_template("check.html")

@app.route("/predict", methods=["POST"])
def predict():
    request_id = uuid.uuid4().hex
    started = time.monotonic()
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    payload = request.get_json(silent=True)
    if payload is None:
        log.warning("[%s] Rejected request: no JSON body", request_id)
        record_blackbox(
            request_id=request_id, client_ip=client_ip, status="rejected",
            reason="Expected a JSON body.", raw_payload=None,
            duration_ms=round((time.monotonic() - started) * 1000, 2),
        )
        return jsonify({"error": "Expected a JSON body."}), 400

    try:
        parsed_features = parse_and_validate(payload)
    except ValidationError as e:
        log.info("[%s] Validation failed: %s", request_id, e)
        record_blackbox(
            request_id=request_id, client_ip=client_ip, status="rejected",
            reason=str(e), raw_payload=payload,
            duration_ms=round((time.monotonic() - started) * 1000, 2),
        )
        return jsonify({"error": str(e)}), 400

    try:
        approved, probability_approved = run_model(parsed_features)
    except Exception:
        log.exception("[%s] Model inference failed", request_id)
        record_blackbox(
            request_id=request_id, client_ip=client_ip, status="error",
            reason="Model inference failed", features=parsed_features,
            duration_ms=round((time.monotonic() - started) * 1000, 2),
        )
        return jsonify({"error": "Prediction failed on the server."}), 500

    if approved:
        message = f"You're likely to be approved — estimated {probability_approved * 100:.1f}% approval confidence."
    else:
        message = f"You're unlikely to be approved right now — estimated {probability_approved * 100:.1f}% approval confidence."

    duration_ms = round((time.monotonic() - started) * 1000, 2)
    log.info(
        "[%s] Prediction complete: approved=%s probability=%.4f (%.1fms)",
        request_id, approved, probability_approved, duration_ms,
    )
    record_blackbox(
        request_id=request_id, client_ip=client_ip,
        status="approved" if approved else "declined",
        features=parsed_features, probability=round(probability_approved, 4),
        duration_ms=duration_ms,
    )

    return jsonify({
        "request_id": request_id,
        "approved": approved,
        "probability": round(probability_approved, 4),
        "score": round(probability_approved * 100, 1),
        "message": message,
    })

if __name__ == "__main__":
    app.run(debug=True)