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
