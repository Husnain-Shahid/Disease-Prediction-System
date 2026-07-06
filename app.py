from flask import Flask, render_template, request, send_file, jsonify
from fpdf import FPDF
from database import init_db, insert_record, fetch_all
from symptoms_questionnaire import (
    SYMPTOM_QUESTIONNAIRE,
    get_symptom_by_category,
    get_all_categories,
    get_questionnaire_fields,
)
import os
import pickle
import re
import logging
import json
import urllib.parse
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DB
init_db()

# Load model
model = pickle.load(open("saved_model/pipeline.pkl", "rb"))
feature_columns = pickle.load(open("saved_model/feature_columns.pkl", "rb"))

# Load questionnaire mapping (category -> list of (field_key, question))
questionnaire_map = get_questionnaire_fields()

# Build mapping from questionnaire field keys -> feature column names
def build_field_to_feature_map(questionnaire_map, feature_columns):
    norm_feature_map = {
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c
        for c in feature_columns
    }

    mapping = {}
    for cat, items in questionnaire_map.items():
        for field_key, question in items:
            fk_norm = re.sub(r"[^a-z0-9]+", "_", field_key.strip().lower()).strip("_")

            # Exact normalized match
            if fk_norm in norm_feature_map:
                mapping[field_key] = norm_feature_map[fk_norm]
                continue

            # Fallback: try the question text itself if it already contains the feature name.
            q_norm = re.sub(r"[^a-z0-9]+", "_", str(question).strip().lower()).strip("_")
            if q_norm in norm_feature_map:
                mapping[field_key] = norm_feature_map[q_norm]

    return mapping


field_to_feature = build_field_to_feature_map(questionnaire_map, feature_columns)

CATEGORY_RULES = [
    (
        "Respiratory & ENT",
        [
            "breath",
            "cough",
            "wheez",
            "sputum",
            "throat",
            "nasal",
            "sinus",
            "nose",
            "ear",
            "hoarse",
            "coryza",
            "sneez",
            "tonsil",
            "voice",
        ],
    ),
    (
        "Cardiac & Circulatory",
        [
            "heart",
            "cardiac",
            "palpitation",
            "heartbeat",
            "blood pressure",
            "circulation",
            "chest pain",
        ],
    ),
    (
        "Gastrointestinal",
        [
            "abdominal",
            "abdomen",
            "stomach",
            "nausea",
            "vomit",
            "diarrhea",
            "constipation",
            "stool",
            "rectal",
            "bowel",
            "heartburn",
            "regurgitation",
            "jaundice",
            "appetite",
        ],
    ),
    (
        "Urinary",
        [
            "urine",
            "urination",
            "bladder",
            "kidney",
            "renal",
            "prostate",
        ],
    ),
    (
        "Reproductive & Breast",
        [
            "vaginal",
            "vulvar",
            "uterine",
            "menstrual",
            "menopause",
            "pregnan",
            "intercourse",
            "sex",
            "penis",
            "scrot",
            "testicle",
            "ejacul",
            "infertility",
            "breast",
        ],
    ),
    (
        "Neurological",
        [
            "headache",
            "dizz",
            "seiz",
            "memory",
            "slur",
            "paresthesia",
            "faint",
            "halluc",
        ],
    ),
    (
        "Musculoskeletal",
        [
            "joint",
            "muscle",
            "bone",
            "back",
            "neck",
            "shoulder",
            "arm",
            "leg",
            "knee",
            "ankle",
            "hip",
            "wrist",
            "elbow",
            "foot",
            "toe",
            "hand",
            "cramp",
            "spasm",
            "stiff",
            "weak",
            "pain",
            "swelling",
        ],
    ),
    (
        "Skin & Hair",
        [
            "skin",
            "rash",
            "lesion",
            "acne",
            "mole",
            "scalp",
            "nail",
            "hair",
            "itch",
            "wound",
            "ulcer",
            "burn",
        ],
    ),
    (
        "Eye & Vision",
        [
            "eye",
            "vision",
            "pupil",
            "eyelid",
            "blind",
            "lacrimation",
            "double vision",
        ],
    ),
    (
        "General & Systemic",
        [
            "fever",
            "chill",
            "fatigue",
            "weight",
            "hot flash",
            "sweat",
            "thirst",
            "appetite",
            "sleep",
        ],
    ),
]

TARGET_COLUMN_CANDIDATES = ["disease", "diseases"]
DISEASE_MAPPING_PATH = "Disease_symptom_and_patient_profile_dataset.csv"
POSITIVE_VALUES = {"1", "yes", "y", "true", "present"}
NEGATIVE_VALUES = {"0", "no", "n", "false", "absent"}

# Minimum normalized score required to accept a rule-based category prediction
CATEGORY_CONFIDENCE_THRESHOLD = 0.5


def normalize_column_name(name):
    return re.sub(r"\s+", " ", str(name).strip().lower())


def category_key(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def match_category(feature_name):
    normalized = str(feature_name).strip().lower()
    for name, keywords in CATEGORY_RULES:
        if any(keyword in normalized for keyword in keywords):
            return name
    return "General & Systemic"


def series_has_positive(series):
    if series.empty:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) > 0).any()
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(POSITIVE_VALUES).any()


def normalize_symptom_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) > 0).astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    values = pd.Series([0] * len(lowered), index=lowered.index)
    values[lowered.isin(POSITIVE_VALUES)] = 1
    values[lowered.isin(NEGATIVE_VALUES)] = 0
    return values


def group_features(columns):
    groups = {name: [] for name, _ in CATEGORY_RULES}
    fallback_category = "General & Systemic"

    for col in columns:
        normalized = str(col).strip().lower()
        matched = False
        for name, keywords in CATEGORY_RULES:
            if any(keyword in normalized for keyword in keywords):
                groups[name].append(col)
                matched = True
                break
        if not matched:
            groups[fallback_category].append(col)

    return [(name, groups[name]) for name, _ in CATEGORY_RULES]


def build_disease_category_map(data_path, columns):
    if not os.path.exists(data_path):
        return {}

    df = pd.read_csv(data_path)
    df.columns = [normalize_column_name(col) for col in df.columns]
    target_col = next(
        (col for col in TARGET_COLUMN_CANDIDATES if col in df.columns), None
    )
    if not target_col:
        return {}, []

    df = df.rename(columns={target_col: "disease"})
    normalized_to_original = {
        normalize_column_name(col): col for col in columns
    }
    symptom_cols = [
        col for col in df.columns
        if col in normalized_to_original and col != "disease"
    ]

    disease_map = {}
    for disease, group in df.groupby("disease"):
        categories = set()
        for col in symptom_cols:
            if series_has_positive(group[col]):
                original_name = normalized_to_original[col]
                categories.add(match_category(original_name))
        if not categories:
            # If no positive symptom columns found for this disease in the
            # mapping CSV, avoid assigning it to every category (causes
            # unrelated diseases to appear everywhere). Instead, map it only
            # to the fallback general/systemic category.
            categories.add("General & Systemic")

        ordered = [name for name, _ in CATEGORY_RULES if name in categories]
        disease_map[str(disease)] = [category_key(name) for name in ordered]

    return disease_map


def build_disease_profiles(data_path, columns):
    if not os.path.exists(data_path):
        return {}

    df = pd.read_csv(data_path)
    df.columns = [normalize_column_name(col) for col in df.columns]
    target_col = next(
        (col for col in TARGET_COLUMN_CANDIDATES if col in df.columns), None
    )
    if not target_col:
        return {}

    df = df.rename(columns={target_col: "disease"})
    normalized_to_original = {
        normalize_column_name(col): col for col in columns
    }
    symptom_cols = [
        col for col in df.columns
        if col in normalized_to_original and col != "disease"
    ]

    for col in symptom_cols:
        df[col] = normalize_symptom_series(df[col])

    profiles = {}
    for disease, group in df.groupby("disease"):
        prevalence = {
            normalized_to_original[col]: float(group[col].mean())
            for col in symptom_cols
        }
        profiles[str(disease)] = prevalence

    return profiles


category_key_map = {name: category_key(name) for name, _ in CATEGORY_RULES}
disease_category_map = build_disease_category_map(
    DISEASE_MAPPING_PATH,
    feature_columns,
)
disease_profiles = build_disease_profiles(
    DISEASE_MAPPING_PATH,
    feature_columns,
)
category_disease_map = {}
for disease, categories in disease_category_map.items():
    for category in categories:
        category_disease_map.setdefault(category, set()).add(disease)


def extract_json_object(text):
    """Extract a JSON object from Gemini output."""
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def fetch_suggestion_from_gemini(disease, selected_category, confidence, top_symptoms):
    """Generate medicine and advice using the Gemini API.

    Expected environment variable:
    - GEMINI_API_KEY: Google AI Studio API key

    Optional environment variable:
    - GEMINI_MODEL: Gemini model name, defaults to gemini-1.5-flash
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    # If user specified a model, try that first. Otherwise try several common models.
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    candidate_models = [env_model] if env_model else [
        "gemini-1.5-flash",
        "gemini-1.0",
        "text-bison-001",
    ]

    symptoms_text = ", ".join(name for name, _score in top_symptoms[:5]) or "none"
    prompt = f"""
You are a medical assistant for a disease prediction app.

Detected disease: {disease}
Selected category: {selected_category}
Confidence: {confidence}%
Top symptoms: {symptoms_text}

Please provide detailed suggestions based on the disease predicted.
Return ONLY valid JSON with this exact structure:
{{
  "medicine": "short medicine suggestion",
  "advice": "short self-care or doctor advice",
  "suggestions": "detailed suggestions based on the disease predicted",
  "note": "optional short note"
}}

Rules:
- Keep answers brief.
- Do not mention that you are an AI.
- If the disease is unclear, suggest consulting a doctor.
- Do not include markdown, code fences, or extra text.
""".strip()

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
        },
    }

    for model_name in candidate_models:
        if not model_name:
            continue
        logger.info("Trying Gemini model: %s", model_name)
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        request_obj = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request_obj, timeout=12) as response:
                raw = response.read().decode("utf-8")
                logger.info("Gemini raw response (truncated) for %s: %s", model_name, raw[:1500])
                data = json.loads(raw)

                candidates = data.get("candidates", [])
                logger.info("Candidates for %s: %s", model_name, len(candidates))
                if not candidates:
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
                logger.info("Candidate text (truncated) for %s: %s", model_name, text[:1500])
                suggestion = extract_json_object(text)

                if isinstance(suggestion, dict):
                    return {
                        "medicine": suggestion.get("medicine", "Consult Doctor"),
                        "advice": suggestion.get("advice", "Consult a doctor for guidance"),
                        "suggestions": suggestion.get("suggestions", "No detailed suggestions provided."),
                        "note": suggestion.get("note", ""),
                        "source": f"gemini:{model_name}",
                    }
        except Exception:
            logger.exception("Gemini request for model %s failed", model_name)

    return None


def fetch_suggestion_from_chatgpt(disease, selected_category, confidence, top_symptoms):
    """Generate medicine and advice using OpenAI Chat completions (ChatGPT).

    Expected environment variables:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_MODEL: optional model name (defaults to gpt-4o-mini)
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    api_url = "https://api.openai.com/v1/chat/completions"

    symptoms_text = ", ".join(name for name, _ in top_symptoms[:5]) or "none"
    prompt = f"""
You are a concise medical assistant.

Detected disease: {disease}
Category: {selected_category}
Confidence: {confidence}%
Top symptoms: {symptoms_text}

Provide detailed suggestions based on the disease predicted.
Return ONLY valid JSON with these keys: medicine, advice, suggestions, note.
Keep answers brief and medically cautious. Do not include extra commentary.
""".strip()

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant. Provide concise treatment suggestions and safety notes."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }

    request_obj = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=12) as response:
            raw = response.read().decode("utf-8")
            logger.info("OpenAI raw response (truncated): %s", raw[:1500])
            data = json.loads(raw)
            choices = data.get("choices", [])
            if not choices:
                return None

            message = choices[0].get("message", {})
            text = message.get("content", "")
            suggestion = extract_json_object(text)

            if isinstance(suggestion, dict):
                return {
                    "medicine": suggestion.get("medicine", "Consult Doctor"),
                    "advice": suggestion.get("advice", "Consult a doctor for guidance"),
                    "suggestions": suggestion.get("suggestions", "No detailed suggestions provided."),
                    "note": suggestion.get("note", ""),
                    "source": f"openai:{model_name}",
                }
    except Exception:
        logger.exception("OpenAI suggestion request failed")

    return None

# Hardcoded local suggestions removed. Gemini or external API will provide
# medicine/advice suggestions. If unavailable, a generic fallback is used.


@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    """Local suggestion API for testing and development.

    Accepts JSON:
    {
        "disease": "...",
        "category": "...",
        "confidence": 82.4,
        "top_symptoms": ["cough", "fever"]
    }
    """
    payload = request.get_json(silent=True) or {}
    disease = str(payload.get("disease", "")).strip()
    category = str(payload.get("category", "")).strip()
    confidence = payload.get("confidence", 0)
    top_symptoms = payload.get("top_symptoms", [])

    # Prefer Gemini/external API suggestions for medicine/advice
    suggestion = fetch_suggestion_from_gemini(disease, category, confidence, [(s, 0) for s in top_symptoms])
    if suggestion:
        suggestion.setdefault("note", f"Generated suggestion for {disease or 'unknown disease'}")
        return jsonify(suggestion)

    # Generic fallback when no external suggestions are available
    return jsonify(
        {
            "medicine": "Consult Doctor",
            "advice": "Please consult a healthcare professional for guidance.",
            "suggestions": "No detailed suggestions available without an API connection.",
            "note": f"No external suggestion available for {disease or 'unknown disease'}.",
            "source": "local_fallback",
        }
    )


def fetch_suggestion_from_api(disease, selected_category, confidence, top_symptoms):
    """Fetch medicine/advice suggestions.

    Expected environment variables:
    - GEMINI_API_KEY: Google AI Studio API key
    - GEMINI_MODEL: optional Gemini model name
    - SUGGESTION_API_URL: optional custom suggestion endpoint
    - SUGGESTION_API_KEY: optional bearer token for the custom endpoint

    The API response may include: medicine, advice, note, source.
    """
    # Prefer Gemini if API key is available, otherwise try ChatGPT, then custom API.
    gemini_suggestion = fetch_suggestion_from_gemini(
        disease=disease,
        selected_category=selected_category,
        confidence=confidence,
        top_symptoms=top_symptoms,
    )
    if gemini_suggestion:
        return gemini_suggestion

    chatgpt_suggestion = fetch_suggestion_from_chatgpt(
        disease=disease,
        selected_category=selected_category,
        confidence=confidence,
        top_symptoms=top_symptoms,
    )
    if chatgpt_suggestion:
        return chatgpt_suggestion

    api_url = os.getenv("SUGGESTION_API_URL", "").strip()
    if not api_url:
        return None

    payload = {
        "disease": disease,
        "category": selected_category,
        "confidence": confidence,
        "top_symptoms": [name for name, _score in top_symptoms[:5]],
    }
    headers = {
        "Content-Type": "application/json",
    }

    api_key = os.getenv("SUGGESTION_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_obj = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=8) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return {
                    "medicine": data.get("medicine", "Consult Doctor"),
                    "advice": data.get("advice", "No advice"),
                    "suggestions": data.get("suggestions", "No detailed suggestions provided."),
                    "note": data.get("note", ""),
                    "source": data.get("source", api_url),
                }
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, ValueError):
        logger.exception("Suggestion API request failed")

    return None

@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_columns=feature_columns,
        feature_groups=group_features(feature_columns),
        category_key_map=category_key_map,
        questionnaire_map=questionnaire_map,
    )

@app.route("/history")
def history():
    data = fetch_all()
    return render_template("history.html", data=data)

@app.route("/predict", methods=["POST"])
def predict():
    def parse_feature_value(value):
        if value is None:
            return 0

        normalized = str(value).strip().lower()
        if normalized in {"yes", "y", "true", "1"}:
            return 1
        if normalized in {"no", "n", "false", "0"}:
            return 0

        try:
            return int(normalized)
        except ValueError:
            pass

        try:
            return float(normalized)
        except ValueError:
            pass

        return value

    input_row = {}
    selected_category = request.form.get("selected_category", "").strip()
    vitals_fields = {
        "patient_name": request.form.get("patient_name", "").strip(),
        "patient_age": request.form.get("patient_age", "").strip(),
        "patient_gender": request.form.get("patient_gender", "").strip(),
        "patient_blood_pressure": request.form.get(
            "patient_blood_pressure", ""
        ).strip(),
        "patient_sugar": request.form.get("patient_sugar", "").strip(),
    }

    # Validate required patient details
    if not vitals_fields["patient_name"]:
        return jsonify({"error": "Patient name is required"}), 400
    if not vitals_fields["patient_age"]:
        return jsonify({"error": "Patient age is required"}), 400
    if not vitals_fields["patient_gender"]:
        return jsonify({"error": "Patient gender is required"}), 400

    # Parse age to integer
    try:
        patient_age = int(vitals_fields["patient_age"])
    except ValueError:
        return jsonify({"error": "Patient age must be a valid number"}), 400

    # Validate age range
    if patient_age < 0 or patient_age > 150:
        return jsonify({"error": "Patient age must be between 0 and 150"}), 400
    # Parse values for model feature columns (direct form fields)
    for col in feature_columns:
        raw_value = request.form.get(col)
        input_row[col] = parse_feature_value(raw_value)

    # Map questionnaire fields into feature columns when possible
    for cat, items in questionnaire_map.items():
        for field_key, _question in items:
            raw_value = request.form.get(field_key)
            val = parse_feature_value(raw_value)
            mapped = field_to_feature.get(field_key)
            if mapped and mapped in feature_columns:
                input_row[mapped] = val
            else:
                # keep questionnaire-only fields too (won't affect model)
                input_row[field_key] = val

    # Log a compact summary of inputs for debugging
    try:
        compact_inputs = {k: v for k, v in input_row.items() if v}
        logger.info("Selected category: %s", selected_category)
        logger.info("Mapped questionnaire -> features (sample): %s", {k: field_to_feature.get(k) for k in list(field_to_feature)[:10]})
        logger.debug("Input row (non-zero values): %s", compact_inputs)
    except Exception:
        logger.exception("Error while logging request inputs")

    for field, raw_value in vitals_fields.items():
        if field in feature_columns:
            input_row[field] = parse_feature_value(raw_value)

    input_df = pd.DataFrame([input_row], columns=feature_columns)

    # Do rule-based category scoring first if a category is selected
    disease = None
    confidence = 0.0
    explanation_details = {}
    used_rule_based = False
    rule_based_top = []

    if selected_category and disease_profiles:
        allowed = category_disease_map.get(selected_category, set())
        category_features = dict(group_features(feature_columns)).get(
            next(
                (
                    name
                    for name, key in category_key_map.items()
                    if key == selected_category
                ),
                "",
            ),
            [],
        )
        if allowed and category_features:
            disease_scores = {}
            disease_symptom_matches = {}

            for candidate in allowed:
                profile = disease_profiles.get(candidate, {})
                score = 0.0
                weight_count = 0
                matched_symptoms = []
                unmatched_symptoms = []

                for col in category_features:
                    if col not in profile:
                        continue
                    weight = profile[col]
                    answer = 1 if input_row.get(col) else 0

                    if answer == 1:
                        score += weight
                        matched_symptoms.append((col.replace("_", " "), weight))
                    else:
                        score += (1 - weight)
                        unmatched_symptoms.append((col.replace("_", " "), weight))

                    weight_count += 1

                if weight_count == 0:
                    continue

                normalized = score / weight_count
                disease_scores[candidate] = normalized
                disease_symptom_matches[candidate] = {
                    "matched": sorted(
                        matched_symptoms, key=lambda x: x[1], reverse=True
                    ),
                    "unmatched": sorted(
                        unmatched_symptoms, key=lambda x: x[1], reverse=True
                    ),
                }

            if disease_scores:
                best_disease = max(disease_scores, key=disease_scores.get)
                best_score = disease_scores[best_disease]

                # Accept category-based result only if it passes threshold
                if best_score >= CATEGORY_CONFIDENCE_THRESHOLD:
                    disease = best_disease
                    confidence = round(best_score * 100, 2)

                    used_rule_based = True
                    # capture top 5 scores for debug/UI
                    rule_based_top = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:5]

                    matched_info = disease_symptom_matches[best_disease]["matched"][:5]
                    explanation_details = {
                        "top_symptoms": matched_info,
                        "score": best_score,
                        "runner_ups": [
                            (name, disease_scores[name])
                            for name in sorted(
                                disease_scores.keys(),
                                key=lambda x: disease_scores[x],
                                reverse=True,
                            )[1:3]
                        ],
                    }

    # If category scoring didn't yield a result, fall back to ML model
    if not disease:
        try:
            disease = model.predict(input_df)[0]
        except Exception:
            disease = "Unknown"

        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(input_df)[0]
                confidence = round(float(np.max(prob)) * 100, 2)
            except Exception:
                confidence = 0.0

    # Log final decision path
    logger.info("Final disease: %s, confidence: %s, used_rule_based: %s", disease, confidence, used_rule_based)

    api_suggestion = fetch_suggestion_from_api(
        disease=disease,
        selected_category=selected_category,
        confidence=confidence,
        top_symptoms=explanation_details.get("top_symptoms", []),
    )

    if api_suggestion:
        medicine = api_suggestion.get("medicine", "Consult Doctor")
        advice = api_suggestion.get("advice", "No advice")
        suggestions = api_suggestion.get("suggestions", "No detailed suggestions provided.")
        suggestion_source = api_suggestion.get("source", "external_api")
        suggestion_note = api_suggestion.get("note", "")
    else:
        medicine = "Consult Doctor"
        advice = "Please consult a healthcare professional for guidance."
        suggestions = "No detailed suggestions available."
        suggestion_source = "local_fallback"
        suggestion_note = f"No external suggestion available for {disease}."

    # Save to DB
    insert_record(
        vitals_fields["patient_name"],
        patient_age,
        vitals_fields["patient_gender"],
        disease,
        confidence,
        medicine,
        advice
    )

    result = f"""
⚠️ High Probability Match: {disease}
Confidence: {confidence}%

Why Our System Identified This:
"""

    if explanation_details and explanation_details.get("top_symptoms"):
        result += "Top Matching Symptoms:\n"
        for symptom, weight in explanation_details["top_symptoms"]:
            result += f"  • {symptom.title()} (Match: {round(weight * 100, 0)}%)\n"

    if explanation_details and explanation_details.get("runner_ups"):
        result += "\nOther Conditions Considered & Ruled Out:\n"
        for runner_up, score in explanation_details["runner_ups"]:
            result += f"  • {runner_up} (Match: {round(score * 100, 0)}%)\n"

    result += f"""

Recommended Treatment & Suggestions:
Medicine: {medicine}
Advice: {advice}
Suggestions: {suggestions}

Note: Please consult a healthcare professional for a definitive diagnosis.
"""
    # Append debug details when running in debug mode
    if app.debug:
        debug_lines = []
        debug_lines.append("\n--- DEBUG INFO ---")
        debug_lines.append(f"Selected category: {selected_category}")
        debug_lines.append(f"Used rule-based scoring: {used_rule_based}")

        if used_rule_based and rule_based_top:
            debug_lines.append("Top rule-based matches:")
            for name, score in rule_based_top:
                debug_lines.append(f"  - {name}: {round(score*100,2)}%")

        # show mapped positive questionnaire inputs
        mapped_pos = []
        try:
            for fk, feat in field_to_feature.items():
                if feat in input_row and input_row.get(feat):
                    mapped_pos.append((fk, feat))
        except Exception:
            pass

        if mapped_pos:
            debug_lines.append("Mapped positive inputs (question_key -> feature):")
            for fk, feat in mapped_pos:
                debug_lines.append(f"  - {fk} -> {feat}")

        result += "\n" + "\n".join(debug_lines)

    prediction_data = {
        "disease": disease,
        "confidence": confidence,
        "medicine": medicine,
        "advice": advice,
        "suggestions": suggestions,
        "suggestion_source": suggestion_source,
        "suggestion_note": suggestion_note,
        "top_symptoms": explanation_details.get("top_symptoms", []),
        "runner_ups": explanation_details.get("runner_ups", []),
        "selected_category": selected_category,
        "used_rule_based": used_rule_based,
        "rule_based_top": rule_based_top,
    }

    return render_template(
        "index.html",
        prediction_text=result,
        feature_columns=feature_columns,
        feature_groups=group_features(feature_columns),
        category_key_map=category_key_map,
        questionnaire_map=questionnaire_map,
        prediction_data=prediction_data,
    )

@app.route("/download")
def download_pdf():
    data = fetch_all()[0]  # latest record

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Disease: {data[1]}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {data[2]}%", ln=True)
    pdf.cell(200, 10, txt=f"Medicines: {data[3]}", ln=True)
    pdf.cell(200, 10, txt=f"Advice: {data[4]}", ln=True)

    file_path = "report.pdf"
    pdf.output(file_path)

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)