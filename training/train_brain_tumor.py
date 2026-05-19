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
