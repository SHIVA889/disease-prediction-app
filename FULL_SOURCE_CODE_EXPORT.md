# Full Source Code Export

This export contains the current project source files.

## File: app.py

```py
import os
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from database.auth_db import (
    AuthError,
    authenticate_user,
    create_user,
    get_user_auth_record,
    get_user_by_id,
    init_auth_table,
)
from database.db import fetch_prediction_history, init_db, save_prediction
from services.prediction_service import (
    PredictionServiceError,
    predict_brain_tumor,
    predict_diabetes,
    predict_heart,
)
from training.config import PROJECT_DISCLAIMER


PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "disease-project-dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

init_db()
init_auth_table()


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if session.get("user_id"):
            return view_function(*args, **kwargs)

        if request.path.startswith("/api/"):
            return jsonify({"error": "Please log in first."}), 401

        return redirect(url_for("login"))

    return wrapped_view


@app.get("/")
def home():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.get("/login")
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("login.html", disclaimer=PROJECT_DISCLAIMER)


@app.post("/login")
def login_post():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    user_record = get_user_auth_record(username)
    user = authenticate_user(username, password)

    if user_record is None:
        session.clear()
        flash("Account not found. Please sign up first before logging in.", "error")
        return redirect(url_for("login"))

    if user is None:
        session.clear()
        flash("Incorrect username or password. Please try again.", "error")
        return redirect(url_for("login"))

    session["user_id"] = user["id"]
    flash("Login successful.", "success")
    return redirect(url_for("dashboard"))


@app.get("/register")
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("register.html", disclaimer=PROJECT_DISCLAIMER)


@app.post("/register")
def register_post():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    confirm_password = request.form.get("confirm_password") or ""

    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect(url_for("register"))

    try:
        user = create_user(username, password)
    except AuthError as error:
        flash(str(error), "error")
        return redirect(url_for("register"))

    session["user_id"] = user["id"]
    flash("Account created successfully.", "success")
    return redirect(url_for("dashboard"))


@app.get("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.get("/dashboard")
@login_required
def dashboard():
    return render_template("index.html", user=current_user(), disclaimer=PROJECT_DISCLAIMER)


@app.post("/api/predict/heart")
@login_required
def heart_prediction():
    payload = request.get_json(force=True, silent=True) or {}

    try:
        result = predict_heart(payload)
        save_prediction(session["user_id"], "heart", payload, result)
        return jsonify(result)
    except PredictionServiceError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/predict/diabetes")
@login_required
def diabetes_prediction():
    payload = request.get_json(force=True, silent=True) or {}

    try:
        result = predict_diabetes(payload)
        save_prediction(session["user_id"], "diabetes", payload, result)
        return jsonify(result)
    except PredictionServiceError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/predict/brain-tumor")
@login_required
def brain_tumor_prediction():
    uploaded_file = request.files.get("scan")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload a brain MRI scan image."}), 400

    try:
        result = predict_brain_tumor(uploaded_file.stream, uploaded_file.filename)
        save_prediction(session["user_id"], "brain_tumor", {"filename": uploaded_file.filename}, result)
        return jsonify(result)
    except PredictionServiceError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


@app.get("/api/predictions/history")
@login_required
def prediction_history():
    return jsonify({"predictions": fetch_prediction_history(session["user_id"], limit=30)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
```

## File: database\auth_db.py

```py
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

from database.db import get_connection


class AuthError(Exception):
    pass


def init_auth_table():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def create_user(username, password):
    username = username.strip()

    if len(username) < 3:
        raise AuthError("Username must be at least 3 characters long.")

    if len(password) < 4:
        raise AuthError("Password must be at least 4 characters long.")

    password_hash = generate_password_hash(password)

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            connection.commit()
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as error:
        raise AuthError("Username already exists. Please choose another one.") from error

    return get_user_by_id(user_id)


def authenticate_user(username, password):
    username = username.strip()
    if not username or not password:
        return None

    row = get_user_auth_record(username)

    if row is None:
        return None

    if not check_password_hash(row["password_hash"], password):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }


def get_user_auth_record(username):
    with get_connection() as connection:
        return connection.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()


def get_user_by_id(user_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }
```

## File: database\db.py

```py
import json
import sqlite3
from datetime import datetime
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "predictions.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _get_table_columns(connection, table_name):
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def init_db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                disease_key TEXT NOT NULL,
                predicted_label INTEGER,
                predicted_label_text TEXT,
                predicted_class TEXT,
                predicted_class_display TEXT,
                confidence REAL,
                confidence_threshold REAL,
                positive_probability REAL,
                negative_probability REAL,
                bounding_box_json TEXT,
                saved_result_image_path TEXT,
                result_text TEXT,
                input_payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        columns = _get_table_columns(connection, "predictions")
        if "user_id" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
        if "confidence_threshold" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN confidence_threshold REAL")
        if "bounding_box_json" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN bounding_box_json TEXT")
        if "saved_result_image_path" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN saved_result_image_path TEXT")

        connection.commit()


def save_prediction(user_id, disease_key, payload, result):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                user_id,
                disease_key,
                predicted_label,
                predicted_label_text,
                predicted_class,
                predicted_class_display,
                confidence,
                confidence_threshold,
                positive_probability,
                negative_probability,
                bounding_box_json,
                saved_result_image_path,
                result_text,
                input_payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                disease_key,
                result.get("predicted_label"),
                result.get("predicted_label_text"),
                result.get("predicted_class"),
                result.get("predicted_class_display"),
                result.get("confidence"),
                result.get("confidence_threshold"),
                result.get("positive_probability"),
                result.get("negative_probability"),
                json.dumps(result.get("bounding_box")) if result.get("bounding_box") is not None else None,
                result.get("saved_result_image_path"),
                result.get("result_text"),
                json.dumps(payload),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()


def fetch_prediction_history(user_id, limit=30):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                disease_key,
                predicted_label,
                predicted_label_text,
                predicted_class,
                predicted_class_display,
                confidence,
                confidence_threshold,
                positive_probability,
                negative_probability,
                bounding_box_json,
                saved_result_image_path,
                result_text,
                input_payload_json,
                created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    history = []
    for row in rows:
        item = dict(row)
        item["input_payload"] = json.loads(item.pop("input_payload_json"))
        item["bounding_box"] = (
            json.loads(item.pop("bounding_box_json"))
            if item.get("bounding_box_json")
            else None
        )
        history.append(item)
    return history
```

## File: services\prediction_service.py

```py
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

from services.symptom_mapper import (
    SymptomMappingError,
    map_diabetes_symptoms_to_features,
    map_heart_symptoms_to_features,
)
from training.config import (
    BRAIN_CLASS_DISPLAY,
    BRAIN_METADATA_PATH,
    BRAIN_MODEL_PATH,
    DIABETES_MODEL_PATH,
    HEART_MODEL_PATH,
    PROJECT_DISCLAIMER,
)
from training.data_preprocessing import preprocess_uploaded_brain_image
from training.model_utils import load_joblib_artifact


class PredictionServiceError(Exception):
    pass


_model_cache = {}
_brain_cnn_cache = None
_brain_metadata_cache = None


def _get_model(cache_key, model_path):
    if cache_key not in _model_cache:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found at {model_path}. Train the model first."
            )
        _model_cache[cache_key] = load_joblib_artifact(model_path)
    return _model_cache[cache_key]


def _get_brain_cnn_model():
    global _brain_cnn_cache

    if _brain_cnn_cache is None:
        if not BRAIN_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Saved brain CNN model not found at {BRAIN_MODEL_PATH}. Train the model first."
            )
        _brain_cnn_cache = tf.keras.models.load_model(BRAIN_MODEL_PATH)

    return _brain_cnn_cache


def _get_brain_metadata():
    global _brain_metadata_cache

    if _brain_metadata_cache is None:
        if not BRAIN_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Saved brain model metadata not found at {BRAIN_METADATA_PATH}. Train the model first."
            )
        _brain_metadata_cache = load_joblib_artifact(BRAIN_METADATA_PATH)

    return _brain_metadata_cache


def _positive_negative_probabilities(model, features_frame):
    probabilities = model.predict_proba(features_frame)[0]
    class_indices = {label: index for index, label in enumerate(model.classes_)}
    positive_index = class_indices.get(1)
    negative_index = class_indices.get(0)

    positive_probability = (
        float(probabilities[positive_index] * 100) if positive_index is not None else 0.0
    )
    negative_probability = (
        float(probabilities[negative_index] * 100) if negative_index is not None else 0.0
    )
    confidence = float(max(probabilities) * 100)
    return positive_probability, negative_probability, confidence


def _encode_image_to_data_url(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def predict_heart(payload):
    try:
        mapped_features = map_heart_symptoms_to_features(payload)
    except SymptomMappingError as error:
        raise PredictionServiceError(str(error)) from error

    features_frame = pd.DataFrame([mapped_features])
    model = _get_model("heart", HEART_MODEL_PATH)

    predicted_label = int(model.predict(features_frame)[0])
    positive_probability, negative_probability, confidence = _positive_negative_probabilities(
        model, features_frame
    )

    disease_detected = predicted_label == 1
    return {
        "disease_key": "heart",
        "predicted_label": predicted_label,
        "predicted_label_text": "Heart disease likely" if disease_detected else "Heart disease unlikely",
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
        "confidence": confidence,
        "result_text": (
            "The model indicates a higher chance of heart disease."
            if disease_detected
            else "The model indicates a lower chance of heart disease."
        ),
        "disclaimer": PROJECT_DISCLAIMER,
    }


def predict_diabetes(payload):
    try:
        mapped_features = map_diabetes_symptoms_to_features(payload)
    except SymptomMappingError as error:
        raise PredictionServiceError(str(error)) from error

    features_frame = pd.DataFrame([mapped_features])
    model = _get_model("diabetes", DIABETES_MODEL_PATH)

    predicted_label = int(model.predict(features_frame)[0])
    positive_probability, negative_probability, confidence = _positive_negative_probabilities(
        model, features_frame
    )

    disease_detected = predicted_label == 1
    return {
        "disease_key": "diabetes",
        "predicted_label": predicted_label,
        "predicted_label_text": "Diabetes likely" if disease_detected else "Diabetes unlikely",
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
        "confidence": confidence,
        "result_text": (
            "The model indicates a higher chance of diabetes."
            if disease_detected
            else "The model indicates a lower chance of diabetes."
        ),
        "disclaimer": PROJECT_DISCLAIMER,
    }


def predict_brain_tumor(file_stream, filename):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise PredictionServiceError(
            "Upload a JPG, JPEG, or PNG image for brain tumor prediction."
        )

    model = _get_brain_cnn_model()
    metadata = _get_brain_metadata()
    image_size = tuple(metadata["image_size"])

    image_bytes = file_stream.read()
    image_array = preprocess_uploaded_brain_image(BytesIO(image_bytes), image_size=image_size)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = metadata["class_names"][predicted_index]
    confidence = float(probabilities[predicted_index] * 100.0)

    display_name = metadata.get("class_display_map", BRAIN_CLASS_DISPLAY).get(
        predicted_class,
        predicted_class,
    )
    tumor_detected = predicted_class != "notumor"

    return {
        "disease_key": "brain_tumor",
        "predicted_class": predicted_class,
        "predicted_class_display": display_name,
        "predicted_label": 1 if tumor_detected else 0,
        "predicted_label_text": "Tumor detected" if tumor_detected else "No tumor detected",
        "confidence": confidence,
        "result_text": f"The uploaded brain MRI scan is classified as {display_name}.",
        "result_image_data_url": _encode_image_to_data_url(image_bytes),
        "localization_note": "This version uses a saved CNN model to classify the MRI image.",
        "disclaimer": PROJECT_DISCLAIMER,
    }
```

## File: services\symptom_mapper.py

```py
from training.config import DIABETES_FEATURES, HEART_FEATURES


class SymptomMappingError(Exception):
    pass


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _number(payload, field_name, default=None):
    value = payload.get(field_name, default)
    if value in (None, ""):
        raise SymptomMappingError(f"Missing required field: {field_name}")

    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise SymptomMappingError(f"Invalid value for {field_name}") from error


def _level(payload, field_name):
    return int(_clamp(round(_number(payload, field_name)), 0, 3))


def map_heart_symptoms_to_features(payload):
    age = int(_clamp(round(_number(payload, "age")), 18, 100))
    sex = int(_clamp(round(_number(payload, "sex")), 0, 1))

    chest_pain = _level(payload, "chest_pain")
    shortness_of_breath = _level(payload, "shortness_of_breath")
    fatigue = _level(payload, "fatigue")
    irregular_heartbeat = _level(payload, "irregular_heartbeat")
    dizziness = _level(payload, "dizziness")
    arm_jaw_pain = _level(payload, "arm_jaw_pain")

    symptom_total = (
        chest_pain
        + shortness_of_breath
        + fatigue
        + irregular_heartbeat
        + dizziness
        + arm_jaw_pain
    )

    if chest_pain >= 2 and arm_jaw_pain >= 1:
        cp = 0
    elif chest_pain >= 2:
        cp = 1
    elif chest_pain >= 1 or arm_jaw_pain >= 2:
        cp = 2
    else:
        cp = 3

    trestbps = int(_clamp(110 + age * 0.35 + shortness_of_breath * 6 + dizziness * 5 + irregular_heartbeat * 4, 90, 220))
    chol = int(_clamp(155 + age * 0.95 + fatigue * 10 + arm_jaw_pain * 12 + irregular_heartbeat * 8, 120, 360))
    fbs = 1 if symptom_total >= 11 else 0

    if irregular_heartbeat >= 3:
        restecg = 2
    elif irregular_heartbeat >= 1 or dizziness >= 2:
        restecg = 1
    else:
        restecg = 0

    thalach = int(_clamp(205 - age - shortness_of_breath * 7 - fatigue * 5 - irregular_heartbeat * 5, 60, 202))
    exang = 1 if shortness_of_breath >= 2 or chest_pain >= 2 else 0
    oldpeak = round(_clamp(0.1 + chest_pain * 0.65 + shortness_of_breath * 0.55 + dizziness * 0.5 + fatigue * 0.25, 0.0, 6.0), 1)

    if dizziness >= 3:
        slope = 2
    elif shortness_of_breath >= 2 or fatigue >= 2:
        slope = 1
    else:
        slope = 0

    ca = int(_clamp(symptom_total // 4, 0, 3))

    if irregular_heartbeat >= 2 or shortness_of_breath >= 2:
        thal = 3
    elif chest_pain >= 2:
        thal = 2
    else:
        thal = 1

    features = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    return {feature: features[feature] for feature in HEART_FEATURES}


def map_diabetes_symptoms_to_features(payload):
    age = int(_clamp(round(_number(payload, "age")), 10, 100))
    pregnancies = int(_clamp(round(_number(payload, "pregnancies", 0)), 0, 15))

    increased_hunger = _level(payload, "increased_hunger")
    frequent_urination = _level(payload, "frequent_urination")
    excessive_thirst = _level(payload, "excessive_thirst")
    unexplained_weight_loss = _level(payload, "unexplained_weight_loss")
    fatigue_tiredness = _level(payload, "fatigue_tiredness")
    blurred_vision = _level(payload, "blurred_vision")

    symptom_total = (
        increased_hunger
        + frequent_urination
        + excessive_thirst
        + unexplained_weight_loss
        + fatigue_tiredness
        + blurred_vision
    )

    glucose = int(_clamp(85 + increased_hunger * 12 + frequent_urination * 15 + excessive_thirst * 14 + unexplained_weight_loss * 10 + blurred_vision * 12, 70, 240))
    blood_pressure = int(_clamp(68 + age * 0.2 + fatigue_tiredness * 3 + blurred_vision * 2, 50, 130))
    skin_thickness = int(_clamp(18 + increased_hunger * 2 + fatigue_tiredness * 2 + frequent_urination, 7, 60))
    insulin = int(_clamp(50 + increased_hunger * 30 + excessive_thirst * 18 + frequent_urination * 22, 15, 320))
    bmi = round(_clamp(21 + increased_hunger * 1.4 + fatigue_tiredness * 0.8 - unexplained_weight_loss * 0.5, 16.0, 45.0), 1)
    diabetes_pedigree_function = round(_clamp(0.2 + (symptom_total / 18) * 1.1, 0.1, 2.5), 3)

    features = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age,
    }

    return {feature: features[feature] for feature in DIABETES_FEATURES}
```

## File: training\config.py

```py
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
METRICS_DIR = MODELS_DIR / "metrics"

HEART_DATASET_PATH = DATASETS_DIR / "heartdatset.csv"
DIABETES_DATASET_PATH = DATASETS_DIR / "diabetes.xls"

BRAIN_ZIP_CANDIDATES = [
    Path(r"C:\Users\ADMIN\AppData\Local\Temp\Epic and CSCR hospital Dataset.zip"),
    Path(r"C:\Users\ADMIN\Downloads\Brain Tumor MRI Dataset (Glioma, Meningioma, Pitui.zip"),
]
BRAIN_EXTRACT_DIR = DATASETS_DIR / "brain_tumor_mri"

HEART_MODEL_PATH = MODELS_DIR / "heart_model.joblib"
DIABETES_MODEL_PATH = MODELS_DIR / "diabetes_model.joblib"
BRAIN_MODEL_PATH = MODELS_DIR / "brain_tumor_cnn.keras"
BRAIN_METADATA_PATH = MODELS_DIR / "brain_tumor_metadata.joblib"

HEART_METRICS_PATH = METRICS_DIR / "heart_metrics.json"
DIABETES_METRICS_PATH = METRICS_DIR / "diabetes_metrics.json"
BRAIN_METRICS_PATH = METRICS_DIR / "brain_tumor_metrics.json"

HEART_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
HEART_TARGET = "target"

DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
DIABETES_TARGET = "Outcome"

BRAIN_IMAGE_SIZE = (96, 96)
BRAIN_BATCH_SIZE = 32
BRAIN_EPOCHS = 12
BRAIN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

BRAIN_CLASS_DISPLAY = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
PROJECT_DISCLAIMER = (
    "This project is for educational and research purposes only and not for clinical diagnosis."
)


def ensure_directories():
    for directory in (
        DATASETS_DIR,
        MODELS_DIR,
        UPLOADS_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
```

## File: training\data_ingestion.py

```py
import shutil
import zipfile
from pathlib import Path

from training.config import (
    BRAIN_EXTRACT_DIR,
    BRAIN_ZIP_CANDIDATES,
    DIABETES_DATASET_PATH,
    HEART_DATASET_PATH,
    ensure_directories,
)


def copy_dataset_if_needed(source_path, destination_path):
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    if destination_path.exists():
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def ensure_tabular_datasets():
    ensure_directories()

    if not HEART_DATASET_PATH.exists():
        default_source = Path(r"C:\Users\ADMIN\Downloads\heartdatset.csv")
        if default_source.exists():
            copy_dataset_if_needed(default_source, HEART_DATASET_PATH)

    if not DIABETES_DATASET_PATH.exists():
        for source in (
            Path(r"C:\Users\ADMIN\Downloads\diabetes.xls"),
            Path(r"C:\Users\ADMIN\Downloads\diabetes(1).xls"),
        ):
            if source.exists():
                copy_dataset_if_needed(source, DIABETES_DATASET_PATH)
                break

    if not HEART_DATASET_PATH.exists():
        raise FileNotFoundError(f"Heart dataset not found at {HEART_DATASET_PATH}.")

    if not DIABETES_DATASET_PATH.exists():
        raise FileNotFoundError(f"Diabetes dataset not found at {DIABETES_DATASET_PATH}.")

    return {
        "heart": HEART_DATASET_PATH,
        "diabetes": DIABETES_DATASET_PATH,
    }


def _extract_zip(zip_path, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination_dir)


def _find_brain_dataset_root(search_dir):
    search_dir = Path(search_dir)

    if (search_dir / "Train").exists() and (search_dir / "Test").exists():
        return search_dir

    for folder in search_dir.rglob("*"):
        if folder.is_dir() and (folder / "Train").exists() and (folder / "Test").exists():
            return folder

    return None


def ensure_brain_dataset():
    ensure_directories()

    existing_root = _find_brain_dataset_root(BRAIN_EXTRACT_DIR)
    if existing_root is not None:
        return existing_root

    zip_source = None
    for candidate in BRAIN_ZIP_CANDIDATES:
        if candidate.exists():
            zip_source = candidate
            break

    if zip_source is None:
        raise FileNotFoundError(
            "Brain tumor MRI zip dataset not found in the expected locations."
        )

    _extract_zip(zip_source, BRAIN_EXTRACT_DIR)

    for nested_zip in BRAIN_EXTRACT_DIR.rglob("*.zip"):
        nested_output = nested_zip.parent / nested_zip.stem
        if not nested_output.exists():
            _extract_zip(nested_zip, nested_output)

    dataset_root = _find_brain_dataset_root(BRAIN_EXTRACT_DIR)
    if dataset_root is None:
        raise FileNotFoundError(
            "Train and Test folders were not found after extracting the brain MRI dataset."
        )

    return dataset_root
```

## File: training\data_preprocessing.py

```py
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

from training.config import (
    BRAIN_BATCH_SIZE,
    BRAIN_IMAGE_SIZE,
    DIABETES_FEATURES,
    DIABETES_TARGET,
    HEART_FEATURES,
    HEART_TARGET,
    RANDOM_STATE,
)


def _expand_single_column_csv(dataframe):
    if len(dataframe.columns) != 1:
        return dataframe

    only_column = dataframe.columns[0]
    if "," not in only_column:
        return dataframe

    header = [column.strip().replace('"', "") for column in only_column.split(",")]
    expanded = (
        dataframe.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.replace('"', "", regex=False)
        .str.split(",", expand=True)
    )
    expanded.columns = header
    return expanded


def load_heart_dataframe(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe = _expand_single_column_csv(dataframe)
    dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
    dataframe = dataframe.dropna().reset_index(drop=True)
    return dataframe


def load_diabetes_dataframe(file_path):
    file_path = Path(file_path)

    try:
        dataframe = pd.read_csv(file_path)
    except Exception:
        dataframe = pd.read_excel(file_path)

    dataframe = _expand_single_column_csv(dataframe)
    dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
    dataframe = dataframe.dropna().reset_index(drop=True)
    return dataframe


def clean_diabetes_dataframe(dataframe):
    dataframe = dataframe.copy()
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for column in zero_as_missing:
        dataframe[column] = dataframe[column].replace(0, np.nan)

    dataframe = dataframe.fillna(dataframe.median(numeric_only=True))
    return dataframe


def split_heart_features_and_target(dataframe):
    features = dataframe[HEART_FEATURES].copy()
    target = dataframe[HEART_TARGET].astype(int)
    return features, target


def split_diabetes_features_and_target(dataframe):
    features = dataframe[DIABETES_FEATURES].copy()
    target = dataframe[DIABETES_TARGET].astype(int)
    return features, target


def preprocess_uploaded_brain_image(image_stream, image_size=BRAIN_IMAGE_SIZE):
    image = Image.open(image_stream)
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize(image_size)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    return np.expand_dims(image_array, axis=0)


def _normalize_brain_batch(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def create_brain_image_dataset(
    directory,
    image_size=BRAIN_IMAGE_SIZE,
    batch_size=BRAIN_BATCH_SIZE,
    shuffle=True,
    validation_split=None,
    subset=None,
):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_split=validation_split,
        subset=subset,
        seed=RANDOM_STATE,
    )

    class_names = list(dataset.class_names)
    dataset = dataset.map(_normalize_brain_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, class_names
```

## File: training\model_utils.py

```py
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score


def save_joblib_artifact(model_object, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_object, file_path)


def load_joblib_artifact(file_path):
    return joblib.load(file_path)


def save_json_artifact(data, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def binary_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def multiclass_metrics(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }
```

## File: training\train_heart.py

```py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from training.config import (
    HEART_DATASET_PATH,
    HEART_METRICS_PATH,
    HEART_MODEL_PATH,
    RANDOM_STATE,
)
from training.data_ingestion import ensure_tabular_datasets
from training.data_preprocessing import (
    load_heart_dataframe,
    split_heart_features_and_target,
)
from training.model_utils import binary_metrics, save_joblib_artifact, save_json_artifact


def train_heart_model():
    ensure_tabular_datasets()

    dataframe = load_heart_dataframe(HEART_DATASET_PATH)
    features, target = split_heart_features_and_target(dataframe)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = binary_metrics(y_test, predictions, probabilities)

    save_joblib_artifact(model, HEART_MODEL_PATH)
    save_json_artifact(metrics, HEART_METRICS_PATH)

    return {
        "model_path": str(HEART_MODEL_PATH),
        "metrics_path": str(HEART_METRICS_PATH),
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = train_heart_model()
    print(f"Heart model saved to: {result['model_path']}")
    print(f"Heart metrics saved to: {result['metrics_path']}")
```

## File: training\train_diabetes.py

```py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.config import (
    DIABETES_DATASET_PATH,
    DIABETES_METRICS_PATH,
    DIABETES_MODEL_PATH,
    RANDOM_STATE,
)
from training.data_ingestion import ensure_tabular_datasets
from training.data_preprocessing import (
    clean_diabetes_dataframe,
    load_diabetes_dataframe,
    split_diabetes_features_and_target,
)
from training.model_utils import binary_metrics, save_joblib_artifact, save_json_artifact


def train_diabetes_model():
    ensure_tabular_datasets()

    dataframe = load_diabetes_dataframe(DIABETES_DATASET_PATH)
    dataframe = clean_diabetes_dataframe(dataframe)
    features, target = split_diabetes_features_and_target(dataframe)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = binary_metrics(y_test, predictions, probabilities)

    save_joblib_artifact(model, DIABETES_MODEL_PATH)
    save_json_artifact(metrics, DIABETES_METRICS_PATH)

    return {
        "model_path": str(DIABETES_MODEL_PATH),
        "metrics_path": str(DIABETES_METRICS_PATH),
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = train_diabetes_model()
    print(f"Diabetes model saved to: {result['model_path']}")
    print(f"Diabetes metrics saved to: {result['metrics_path']}")
```

## File: training\train_brain_tumor.py

```py
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import (
    BRAIN_BATCH_SIZE,
    BRAIN_CLASS_DISPLAY,
    BRAIN_EPOCHS,
    BRAIN_IMAGE_SIZE,
    BRAIN_METADATA_PATH,
    BRAIN_METRICS_PATH,
    BRAIN_MODEL_PATH,
    RANDOM_STATE,
)
from training.data_ingestion import ensure_brain_dataset
from training.data_preprocessing import create_brain_image_dataset
from training.model_utils import multiclass_metrics, save_joblib_artifact, save_json_artifact


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_brain_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _collect_true_labels(dataset):
    return np.concatenate([labels.numpy() for _, labels in dataset], axis=0)


def train_brain_tumor_model():
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    dataset_root = ensure_brain_dataset()

    train_dataset, class_names = create_brain_image_dataset(
        dataset_root / "Train",
        image_size=BRAIN_IMAGE_SIZE,
        batch_size=BRAIN_BATCH_SIZE,
        validation_split=0.2,
        subset="training",
    )
    validation_dataset, _ = create_brain_image_dataset(
        dataset_root / "Train",
        image_size=BRAIN_IMAGE_SIZE,
        batch_size=BRAIN_BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        shuffle=False,
    )
    test_dataset, _ = create_brain_image_dataset(
        dataset_root / "Test",
        image_size=BRAIN_IMAGE_SIZE,
        batch_size=BRAIN_BATCH_SIZE,
        shuffle=False,
    )

    model = build_brain_cnn_model(
        input_shape=(BRAIN_IMAGE_SIZE[0], BRAIN_IMAGE_SIZE[1], 1),
        num_classes=len(class_names),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BRAIN_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=BRAIN_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(BRAIN_MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_dataset, verbose=0)

    probabilities = best_model.predict(test_dataset, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = _collect_true_labels(test_dataset)
    label_names = np.array(class_names)[true_labels]
    prediction_names = np.array(class_names)[predictions]

    metrics = multiclass_metrics(label_names, prediction_names)
    metrics["test_loss"] = round(float(test_loss), 4)
    metrics["test_accuracy"] = round(float(test_accuracy), 4)
    metrics["dataset_root"] = str(dataset_root)
    metrics["image_size"] = list(BRAIN_IMAGE_SIZE)
    metrics["batch_size"] = BRAIN_BATCH_SIZE
    metrics["epochs_requested"] = BRAIN_EPOCHS
    metrics["epochs_trained"] = len(history.history["loss"])
    metrics["classes"] = class_names
    metrics["history"] = {
        "accuracy": [round(float(value), 4) for value in history.history.get("accuracy", [])],
        "val_accuracy": [round(float(value), 4) for value in history.history.get("val_accuracy", [])],
        "loss": [round(float(value), 4) for value in history.history.get("loss", [])],
        "val_loss": [round(float(value), 4) for value in history.history.get("val_loss", [])],
    }

    metadata = {
        "class_names": class_names,
        "image_size": list(BRAIN_IMAGE_SIZE),
        "class_display_map": BRAIN_CLASS_DISPLAY,
        "model_type": "cnn",
        "dataset_root": str(dataset_root),
    }

    save_joblib_artifact(metadata, BRAIN_METADATA_PATH)
    save_json_artifact(metrics, BRAIN_METRICS_PATH)

    return {
        "model_path": str(BRAIN_MODEL_PATH),
        "metadata_path": str(BRAIN_METADATA_PATH),
        "metrics_path": str(BRAIN_METRICS_PATH),
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = train_brain_tumor_model()
    print(f"Brain tumor CNN model saved to: {result['model_path']}")
    print(f"Brain tumor metadata saved to: {result['metadata_path']}")
    print(f"Brain tumor metrics saved to: {result['metrics_path']}")
```

## File: training\train_all.py

```py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import ensure_directories
from training.train_brain_tumor import train_brain_tumor_model
from training.train_diabetes import train_diabetes_model
from training.train_heart import train_heart_model


def main():
    ensure_directories()

    heart_result = train_heart_model()
    diabetes_result = train_diabetes_model()
    brain_result = train_brain_tumor_model()

    print("Training completed successfully.")
    print(heart_result["model_path"])
    print(diabetes_result["model_path"])
    print(brain_result["model_path"])
    print(brain_result["metadata_path"])


if __name__ == "__main__":
    main()
```

## File: templates\index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Disease Prediction System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}" />
  </head>
  <body>
    <div class="app-shell">
      <aside class="sidebar">
        <div class="brand">
          <h1>Disease Prediction System</h1>
        </div>

        <div class="user-box">
          <span>Logged in as</span>
          <strong>{{ user.username }}</strong>
        </div>

        <nav id="diseaseNav" class="nav-list" aria-label="Disease navigation"></nav>

        <a class="secondary-button link-button" href="{{ url_for('logout') }}">Log Out</a>

        <div class="disclaimer-box sidebar-disclaimer">
          <strong>Disclaimer</strong>
          <p>{{ disclaimer }}</p>
        </div>
      </aside>

      <main class="content">
        <section id="predictionPanel" class="panel"></section>

        <section class="panel history-panel">
          <div class="history-header">
            <h3>Prediction History</h3>
            <button id="refreshHistoryButton" class="secondary-button" type="button">
              Refresh History
            </button>
          </div>

          <div id="historyList" class="history-list"></div>
        </section>
      </main>
    </div>

    <script src="{{ url_for('static', filename='script.js') }}"></script>
  </body>
</html>
```

## File: templates\login.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Log In</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}" />
  </head>
  <body class="auth-body">
    <div class="auth-shell">
      <div class="auth-card">
        <h1>Log In</h1>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            <div class="message-stack">
              {% for category, message in messages %}
                <div class="flash-message {{ category }}">{{ message }}</div>
              {% endfor %}
            </div>
          {% endif %}
        {% endwith %}

        <form class="auth-form" method="post" action="{{ url_for('login_post') }}">
          <label class="input-block">
            <span>Username</span>
            <input type="text" name="username" required />
          </label>

          <label class="input-block">
            <span>Password</span>
            <input type="password" name="password" required />
          </label>

          <button class="primary-button auth-button" type="submit">Log In</button>
        </form>

        <p class="auth-switch">
          Don't have an account?
          <a href="{{ url_for('register') }}">Sign Up</a>
        </p>

        <div class="disclaimer-box auth-disclaimer">
          <strong>Disclaimer</strong>
          <p>{{ disclaimer }}</p>
        </div>
      </div>
    </div>
  </body>
</html>
```

## File: templates\register.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Sign Up</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}" />
  </head>
  <body class="auth-body">
    <div class="auth-shell">
      <div class="auth-card">
        <h1>Sign Up</h1>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            <div class="message-stack">
              {% for category, message in messages %}
                <div class="flash-message {{ category }}">{{ message }}</div>
              {% endfor %}
            </div>
          {% endif %}
        {% endwith %}

        <form class="auth-form" method="post" action="{{ url_for('register_post') }}">
          <label class="input-block">
            <span>Username</span>
            <input type="text" name="username" required />
          </label>

          <label class="input-block">
            <span>Password</span>
            <input type="password" name="password" required />
          </label>

          <label class="input-block">
            <span>Confirm Password</span>
            <input type="password" name="confirm_password" required />
          </label>

          <button class="primary-button auth-button" type="submit">Create Account</button>
        </form>

        <p class="auth-switch">
          Already have an account?
          <a href="{{ url_for('login') }}">Log In</a>
        </p>

        <div class="disclaimer-box auth-disclaimer">
          <strong>Disclaimer</strong>
          <p>{{ disclaimer }}</p>
        </div>
      </div>
    </div>
  </body>
</html>
```

## File: static\script.js

```js
const SEVERITY_OPTIONS = [
  { value: 0, label: "None" },
  { value: 1, label: "Mild" },
  { value: 2, label: "Moderate" },
  { value: 3, label: "Severe" }
];

const DISEASES = [
  {
    id: "brain_tumor",
    title: "Brain Tumor",
    navLabel: "Brain Tumor Prediction",
    mode: "image",
    endpoint: "/api/predict/brain-tumor"
  },
  {
    id: "heart",
    title: "Heart Disease",
    navLabel: "Heart Disease Prediction",
    mode: "form",
    endpoint: "/api/predict/heart",
    fields: [
      { key: "age", label: "Age", type: "number", step: "1" },
      {
        key: "sex",
        label: "Sex",
        type: "select",
        options: [
          { value: 0, label: "Female" },
          { value: 1, label: "Male" }
        ]
      },
      { key: "chest_pain", label: "Chest Pain", type: "select", options: SEVERITY_OPTIONS },
      {
        key: "shortness_of_breath",
        label: "Shortness of Breath",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      { key: "fatigue", label: "Fatigue", type: "select", options: SEVERITY_OPTIONS },
      {
        key: "irregular_heartbeat",
        label: "Irregular Heartbeat / Palpitations",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "dizziness",
        label: "Dizziness or Fainting",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "arm_jaw_pain",
        label: "Pain in Left Arm, Shoulder, or Jaw",
        type: "select",
        options: SEVERITY_OPTIONS
      }
    ]
  },
  {
    id: "diabetes",
    title: "Diabetes",
    navLabel: "Diabetes Prediction",
    mode: "form",
    endpoint: "/api/predict/diabetes",
    fields: [
      { key: "age", label: "Age", type: "number", step: "1" },
      {
        key: "pregnancies",
        label: "Pregnancies (If Applicable)",
        type: "number",
        step: "1"
      },
      {
        key: "increased_hunger",
        label: "Increased Hunger",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "frequent_urination",
        label: "Frequent Urination",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "excessive_thirst",
        label: "Excessive Thirst",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "unexplained_weight_loss",
        label: "Unexplained Weight Loss",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "fatigue_tiredness",
        label: "Fatigue / Tiredness",
        type: "select",
        options: SEVERITY_OPTIONS
      },
      {
        key: "blurred_vision",
        label: "Blurred Vision",
        type: "select",
        options: SEVERITY_OPTIONS
      }
    ]
  }
];

let activeDiseaseId = DISEASES[0].id;

const diseaseNav = document.getElementById("diseaseNav");
const predictionPanel = document.getElementById("predictionPanel");
const historyList = document.getElementById("historyList");
const refreshHistoryButton = document.getElementById("refreshHistoryButton");

function getActiveDisease() {
  return DISEASES.find((disease) => disease.id === activeDiseaseId);
}

function renderNav() {
  diseaseNav.innerHTML = DISEASES.map(
    (disease) => `
      <button
        type="button"
        class="nav-button ${disease.id === activeDiseaseId ? "active" : ""}"
        data-disease="${disease.id}"
      >
        ${escapeHtml(disease.navLabel)}
      </button>
    `
  ).join("");
}

function renderPredictionPanel() {
  const disease = getActiveDisease();

  predictionPanel.innerHTML = `
    <div class="section-header compact-header">
      <h2>${escapeHtml(disease.title)} Prediction</h2>
    </div>

    ${disease.mode === "image" ? renderBrainUploadForm() : renderTabularForm(disease)}

    <div id="resultCard" class="result-card empty">
      <div class="result-title">Prediction Result</div>
      <p class="result-summary">Submit the form to see the prediction result.</p>
    </div>
  `;

  attachPredictionHandler(disease);
}

function renderBrainUploadForm() {
  return `
    <form id="predictionForm">
      <div class="upload-block">
        <label class="input-block">
          <span>Choose Brain MRI Scan</span>
          <input id="brainScanInput" name="scan" type="file" accept="image/*" required />
        </label>

        <img id="brainPreview" class="upload-preview" alt="Brain MRI preview" />
      </div>

      <div class="button-row">
        <button class="primary-button" type="submit">Predict Brain Tumor</button>
        <button class="secondary-button" type="reset">Reset</button>
      </div>
    </form>
  `;
}

function renderTabularForm(disease) {
  return `
    <form id="predictionForm">
      <div class="form-grid">
        ${disease.fields.map(renderField).join("")}
      </div>

      <div class="button-row">
        <button class="primary-button" type="submit">Predict ${escapeHtml(disease.title)}</button>
        <button class="secondary-button" type="reset">Reset</button>
      </div>
    </form>
  `;
}

function renderField(field) {
  if (field.type === "select") {
    return `
      <label class="input-block">
        <span>${escapeHtml(field.label)}</span>
        <select name="${escapeHtml(field.key)}" required>
          <option value="">Select</option>
          ${field.options
            .map(
              (option) =>
                `<option value="${escapeHtml(String(option.value))}">${escapeHtml(option.label)}</option>`
            )
            .join("")}
        </select>
      </label>
    `;
  }

  return `
    <label class="input-block">
      <span>${escapeHtml(field.label)}</span>
      <input
        type="number"
        name="${escapeHtml(field.key)}"
        step="${escapeHtml(field.step || "1")}"
        ${field.key === "pregnancies" ? 'value="0"' : ""}
        required
      />
    </label>
  `;
}

function attachPredictionHandler(disease) {
  const form = document.getElementById("predictionForm");
  const resultCard = document.getElementById("resultCard");

  if (disease.mode === "image") {
    const fileInput = document.getElementById("brainScanInput");
    const preview = document.getElementById("brainPreview");

    fileInput.addEventListener("change", () => {
      const [file] = fileInput.files;
      if (!file) {
        preview.style.display = "none";
        preview.removeAttribute("src");
        return;
      }

      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResultLoading(resultCard, disease.title);

    try {
      const result =
        disease.mode === "image"
          ? await submitBrainTumorPrediction(form, disease)
          : await submitTabularPrediction(form, disease);

      renderPredictionResult(resultCard, result);
      await loadHistory();
    } catch (error) {
      renderError(resultCard, error.message || "Prediction request failed.");
    }
  });

  form.addEventListener("reset", () => {
    window.setTimeout(() => {
      resultCard.className = "result-card empty";
      resultCard.innerHTML = `
        <div class="result-title">Prediction Result</div>
        <p class="result-summary">Submit the form to see the prediction result.</p>
      `;

      const preview = document.getElementById("brainPreview");
      if (preview) {
        preview.style.display = "none";
        preview.removeAttribute("src");
      }
    }, 0);
  });
}

async function submitTabularPrediction(form, disease) {
  const formData = new FormData(form);
  const payload = {};

  disease.fields.forEach((field) => {
    payload[field.key] =
      field.type === "number" ? Number(formData.get(field.key)) : Number(formData.get(field.key));
  });

  const response = await fetch(disease.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  return handleJsonResponse(response);
}

async function submitBrainTumorPrediction(form, disease) {
  const formData = new FormData(form);
  const file = formData.get("scan");

  if (!file || !file.name) {
    throw new Error("Please upload a brain MRI image first.");
  }

  const uploadData = new FormData();
  uploadData.append("scan", file);

  const response = await fetch(disease.endpoint, {
    method: "POST",
    body: uploadData
  });

  return handleJsonResponse(response);
}

async function handleJsonResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (response.status === 401) {
    window.location.href = "/login";
    throw new Error("Please log in again.");
  }

  if (!response.ok) {
    throw new Error(data.error || `Request failed with status ${response.status}.`);
  }

  return data;
}

function setResultLoading(resultCard, title) {
  resultCard.className = "result-card empty";
  resultCard.innerHTML = `
    <div class="result-title">Prediction Result</div>
    <p class="result-summary">Running ${escapeHtml(title.toLowerCase())} prediction...</p>
  `;
}

function renderPredictionResult(resultCard, result) {
  const details = buildResultDetails(result);
  const imageBlock = result.result_image_data_url
    ? `
        <div class="result-image-block">
          <img class="result-image" src="${result.result_image_data_url}" alt="Brain MRI result" />
          ${result.localization_note ? `<p class="helper-text">${escapeHtml(result.localization_note)}</p>` : ""}
        </div>
      `
    : "";
  const disclaimerBlock = result.disclaimer
    ? `<p class="helper-text">${escapeHtml(result.disclaimer)}</p>`
    : "";

  resultCard.className = "result-card";
  resultCard.innerHTML = `
    <div class="result-title">Prediction Result</div>
    <p class="result-summary">${escapeHtml(result.result_text)}</p>
    <div class="detail-grid">
      ${details
        .map(
          (item) => `
            <div class="detail-card">
              <strong>${escapeHtml(item.label)}</strong>
              <p>${escapeHtml(item.value)}</p>
            </div>
          `
        )
        .join("")}
    </div>
    ${imageBlock}
    ${disclaimerBlock}
  `;
}

function renderError(resultCard, message) {
  resultCard.className = "result-card error";
  resultCard.innerHTML = `
    <div class="result-title">Prediction Result</div>
    <p class="result-summary">${escapeHtml(message)}</p>
  `;
}

function buildResultDetails(result) {
  const details = [];

  if (result.predicted_class_display) {
    details.push({ label: "Predicted Class", value: result.predicted_class_display });
  }

  if (result.predicted_label_text) {
    details.push({ label: "Prediction", value: result.predicted_label_text });
  }

  if (typeof result.confidence === "number") {
    details.push({ label: "Confidence", value: `${result.confidence.toFixed(2)}%` });
  }

  if (result.positive_probability !== undefined) {
    details.push({
      label: "Positive Probability",
      value: `${Number(result.positive_probability).toFixed(2)}%`
    });
  }

  if (result.negative_probability !== undefined) {
    details.push({
      label: "Negative Probability",
      value: `${Number(result.negative_probability).toFixed(2)}%`
    });
  }

  return details;
}

async function loadHistory() {
  if (!historyList) {
    return;
  }

  historyList.innerHTML = `<div class="history-empty">Loading history...</div>`;

  try {
    const response = await fetch("/api/predictions/history");
    const data = await handleJsonResponse(response);
    renderHistory(data.predictions || []);
  } catch (error) {
    historyList.innerHTML = `<div class="history-empty">${escapeHtml(error.message || "Could not load history.")}</div>`;
  }
}

function renderHistory(predictions) {
  if (!predictions.length) {
    historyList.innerHTML = `<div class="history-empty">No prediction history yet.</div>`;
    return;
  }

  historyList.innerHTML = predictions
    .map((prediction) => {
      const diseaseName = formatDiseaseName(prediction.disease_key);
      const confidence =
        typeof prediction.confidence === "number"
          ? `${Number(prediction.confidence).toFixed(2)}%`
          : "Not available";

      return `
        <article class="history-item">
          <div class="history-meta-row">
            <strong>${escapeHtml(diseaseName)}</strong>
            <span>${escapeHtml(formatDate(prediction.created_at))}</span>
          </div>
          <p>${escapeHtml(prediction.result_text || "No result text available.")}</p>
          <div class="history-submeta">
            <span>${escapeHtml(prediction.predicted_label_text || prediction.predicted_class_display || "Prediction recorded")}</span>
            <span>Confidence: ${escapeHtml(confidence)}</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function formatDiseaseName(value) {
  if (value === "brain_tumor") {
    return "Brain Tumor";
  }
  if (value === "heart") {
    return "Heart Disease";
  }
  if (value === "diabetes") {
    return "Diabetes";
  }
  return value;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

diseaseNav.addEventListener("click", (event) => {
  const button = event.target.closest("[data-disease]");
  if (!button) {
    return;
  }

  activeDiseaseId = button.dataset.disease;
  renderNav();
  renderPredictionPanel();
});

if (refreshHistoryButton) {
  refreshHistoryButton.addEventListener("click", loadHistory);
}

renderNav();
renderPredictionPanel();
loadHistory();
```

## File: static\styles.css

```css
:root {
  --background: #f4f4f4;
  --sidebar: #ececec;
  --panel: #ffffff;
  --input: #f8f8f8;
  --border: #cccccc;
  --text: #222222;
  --muted: #666666;
  --button: #222222;
  --button-text: #ffffff;
  --soft: #f2f2f2;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  color: var(--text);
  background: var(--background);
}

h1,
h2,
h3,
h4,
p {
  margin: 0;
}

button,
input,
select {
  font: inherit;
}

.app-shell {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  min-height: 100vh;
}

.sidebar {
  padding: 24px 16px;
  background: var(--sidebar);
  border-right: 1px solid var(--border);
}

.brand,
.user-box,
.disclaimer-box,
.panel,
.auth-card {
  border: 1px solid var(--border);
  background: var(--panel);
}

.brand,
.user-box {
  padding: 16px;
}

.brand h1 {
  font-size: 1.35rem;
  line-height: 1.3;
}

.user-box {
  display: grid;
  gap: 6px;
  margin-top: 14px;
}

.user-box span,
.helper-text,
.history-submeta,
.auth-switch,
.disclaimer-box p {
  color: var(--muted);
}

.nav-list {
  display: grid;
  gap: 10px;
  margin: 18px 0;
}

.nav-button,
.primary-button,
.secondary-button,
.auth-button {
  border-radius: 0;
}

.nav-button {
  width: 100%;
  padding: 12px 14px;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  text-align: left;
  cursor: pointer;
}

.nav-button.active,
.primary-button {
  background: var(--button);
  color: var(--button-text);
}

.secondary-button {
  background: var(--panel);
  color: var(--text);
}

.primary-button,
.secondary-button {
  padding: 12px 18px;
  border: 1px solid var(--border);
  cursor: pointer;
}

.primary-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.link-button {
  display: inline-flex;
  justify-content: center;
  text-decoration: none;
}

.disclaimer-box {
  margin-top: 18px;
  padding: 14px;
}

.disclaimer-box strong {
  display: block;
  margin-bottom: 8px;
}

.content {
  padding: 24px;
  display: grid;
  gap: 20px;
}

.panel {
  padding: 20px;
}

.compact-header {
  margin-bottom: 18px;
}

.compact-header h2 {
  font-size: clamp(1.8rem, 3vw, 2.4rem);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.input-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.input-block span {
  font-size: 0.95rem;
  font-weight: 600;
}

.input-block input,
.input-block select {
  min-height: 44px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  background: var(--input);
  color: var(--text);
}

.input-block input:focus,
.input-block select:focus {
  outline: 1px solid var(--text);
}

.upload-block {
  max-width: 640px;
  padding: 20px;
  border: 1px solid var(--border);
  background: var(--soft);
}

.upload-preview {
  display: none;
  width: min(100%, 320px);
  margin-top: 18px;
  border: 1px solid var(--border);
}

.button-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 18px;
}

.result-card {
  margin-top: 20px;
  padding: 16px;
  border: 1px solid var(--border);
  background: var(--panel);
}

.result-card.empty,
.history-item,
.history-empty,
.flash-message {
  background: var(--soft);
}

.result-card.error {
  border-color: #999999;
  background: #eeeeee;
}

.result-title {
  font-size: 1.05rem;
  font-weight: 700;
}

.result-summary,
.history-item p,
.helper-text {
  margin-top: 8px;
  line-height: 1.6;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.detail-card {
  padding: 12px;
  border: 1px solid var(--border);
  background: var(--soft);
}

.result-image-block {
  margin-top: 18px;
}

.result-image {
  display: block;
  width: min(100%, 460px);
  border: 1px solid var(--border);
}

.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 18px;
}

.history-list {
  display: grid;
  gap: 12px;
}

.history-item {
  padding: 14px;
  border: 1px solid var(--border);
}

.history-meta-row,
.history-submeta {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 10px;
}

.history-empty {
  padding: 18px;
  border: 1px solid var(--border);
}

.auth-body {
  display: grid;
  place-items: center;
  padding: 24px;
}

.auth-shell {
  width: min(100%, 440px);
}

.auth-card {
  padding: 28px;
}

.auth-card h1 {
  margin-bottom: 18px;
}

.auth-form {
  display: grid;
  gap: 14px;
}

.auth-button {
  justify-content: center;
}

.auth-switch {
  margin-top: 18px;
}

.auth-switch a {
  color: var(--text);
  text-decoration: underline;
}

.auth-disclaimer {
  margin-top: 18px;
}

.message-stack {
  display: grid;
  gap: 10px;
  margin-bottom: 16px;
}

.flash-message {
  padding: 12px 14px;
  border: 1px solid var(--border);
}

@media (max-width: 1100px) {
  .form-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .app-shell {
    grid-template-columns: 1fr;
  }

  .sidebar {
    border-right: none;
    border-bottom: 1px solid var(--border);
  }

  .content {
    padding: 18px 14px 24px;
  }
}

@media (max-width: 700px) {
  .form-grid,
  .detail-grid {
    grid-template-columns: 1fr;
  }

  .button-row,
  .history-header,
  .history-meta-row,
  .history-submeta {
    flex-direction: column;
    align-items: stretch;
  }
}
```

## File: README.md

```md
# Multiple Disease Prediction System

This project is for educational and research purposes only and not for clinical diagnosis.

## What the project does

- Predicts heart disease likelihood from symptom-based inputs mapped to the heart dataset.
- Predicts diabetes likelihood from symptom-based inputs mapped to the diabetes dataset.
- Classifies brain MRI scans into tumor classes using a saved CNN model.
- Stores user accounts and prediction history in local SQLite databases.

## Current project structure

```text
Disease_project/
|-- app.py
|-- database/
|   |-- auth_db.py
|   |-- db.py
|   `-- predictions.db
|-- datasets/
|   |-- diabetes.xls
|   |-- heartdatset.csv
|   `-- brain_tumor_mri/
|-- models/
|   |-- heart_model.joblib
|   |-- diabetes_model.joblib
|   |-- brain_tumor_cnn.keras
|   |-- brain_tumor_metadata.joblib
|   `-- metrics/
|-- services/
|   |-- prediction_service.py
|   `-- symptom_mapper.py
|-- static/
|   |-- script.js
|   `-- styles.css
|-- templates/
|   |-- index.html
|   |-- login.html
|   `-- register.html
|-- training/
|   |-- config.py
|   |-- data_ingestion.py
|   |-- data_preprocessing.py
|   |-- model_utils.py
|   |-- train_heart.py
|   |-- train_diabetes.py
|   |-- train_brain_tumor.py
|   `-- train_all.py
`-- uploads/
```

## Brain tumor model

The brain tumor feature now uses a simple CNN built with TensorFlow/Keras.

- The CNN is trained once from the MRI image folders.
- The trained model is saved to `models/brain_tumor_cnn.keras`.
- Extra class metadata is saved to `models/brain_tumor_metadata.joblib`.
- The Flask app only loads these saved files during prediction.

## Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Train saved models

Run this once to train and save all three models:

```powershell
python -m training.train_all
```

## Run the Flask app

```powershell
python app.py
```

Then open:

- [http://127.0.0.1:5000/login](http://127.0.0.1:5000/login)

## Saved prediction behavior

- Heart and diabetes use saved `joblib` models.
- Brain tumor prediction uses a saved CNN `.keras` model.
- Prediction history is stored locally in SQLite.
```

## File: requirements.txt

```txt
Flask>=3.1.0
joblib>=1.5.0
numpy>=2.3.0
pandas>=2.3.0
Pillow>=11.0.0
scikit-learn>=1.7.0
tensorflow>=2.21.0
```


