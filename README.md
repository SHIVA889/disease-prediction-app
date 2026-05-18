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
