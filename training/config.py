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
