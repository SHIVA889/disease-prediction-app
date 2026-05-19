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
