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
