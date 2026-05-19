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
