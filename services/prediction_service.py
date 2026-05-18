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
