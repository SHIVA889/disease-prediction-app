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
