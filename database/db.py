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
