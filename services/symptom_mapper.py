from training.config import DIABETES_FEATURES, HEART_FEATURES


class SymptomMappingError(Exception):
    pass


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _number(payload, field_name, default=None):
    value = payload.get(field_name, default)
    if value in (None, ""):
        raise SymptomMappingError(f"Missing required field: {field_name}")

    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise SymptomMappingError(f"Invalid value for {field_name}") from error


def _level(payload, field_name):
    return int(_clamp(round(_number(payload, field_name)), 0, 3))


def map_heart_symptoms_to_features(payload):
    age = int(_clamp(round(_number(payload, "age")), 18, 100))
    sex = int(_clamp(round(_number(payload, "sex")), 0, 1))

    chest_pain = _level(payload, "chest_pain")
    shortness_of_breath = _level(payload, "shortness_of_breath")
    fatigue = _level(payload, "fatigue")
    irregular_heartbeat = _level(payload, "irregular_heartbeat")
    dizziness = _level(payload, "dizziness")
    arm_jaw_pain = _level(payload, "arm_jaw_pain")

    symptom_total = (
        chest_pain
        + shortness_of_breath
        + fatigue
        + irregular_heartbeat
        + dizziness
        + arm_jaw_pain
    )

    if chest_pain >= 2 and arm_jaw_pain >= 1:
        cp = 0
    elif chest_pain >= 2:
        cp = 1
    elif chest_pain >= 1 or arm_jaw_pain >= 2:
        cp = 2
    else:
        cp = 3

    trestbps = int(_clamp(110 + age * 0.35 + shortness_of_breath * 6 + dizziness * 5 + irregular_heartbeat * 4, 90, 220))
    chol = int(_clamp(155 + age * 0.95 + fatigue * 10 + arm_jaw_pain * 12 + irregular_heartbeat * 8, 120, 360))
    fbs = 1 if symptom_total >= 11 else 0

    if irregular_heartbeat >= 3:
        restecg = 2
    elif irregular_heartbeat >= 1 or dizziness >= 2:
        restecg = 1
    else:
        restecg = 0

    thalach = int(_clamp(205 - age - shortness_of_breath * 7 - fatigue * 5 - irregular_heartbeat * 5, 60, 202))
    exang = 1 if shortness_of_breath >= 2 or chest_pain >= 2 else 0
    oldpeak = round(_clamp(0.1 + chest_pain * 0.65 + shortness_of_breath * 0.55 + dizziness * 0.5 + fatigue * 0.25, 0.0, 6.0), 1)

    if dizziness >= 3:
        slope = 2
    elif shortness_of_breath >= 2 or fatigue >= 2:
        slope = 1
    else:
        slope = 0

    ca = int(_clamp(symptom_total // 4, 0, 3))

    if irregular_heartbeat >= 2 or shortness_of_breath >= 2:
        thal = 3
    elif chest_pain >= 2:
        thal = 2
    else:
        thal = 1

    features = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    return {feature: features[feature] for feature in HEART_FEATURES}


def map_diabetes_symptoms_to_features(payload):
    age = int(_clamp(round(_number(payload, "age")), 10, 100))
    pregnancies = int(_clamp(round(_number(payload, "pregnancies", 0)), 0, 15))

    increased_hunger = _level(payload, "increased_hunger")
    frequent_urination = _level(payload, "frequent_urination")
    excessive_thirst = _level(payload, "excessive_thirst")
    unexplained_weight_loss = _level(payload, "unexplained_weight_loss")
    fatigue_tiredness = _level(payload, "fatigue_tiredness")
    blurred_vision = _level(payload, "blurred_vision")

    symptom_total = (
        increased_hunger
        + frequent_urination
        + excessive_thirst
        + unexplained_weight_loss
        + fatigue_tiredness
        + blurred_vision
    )

    glucose = int(_clamp(85 + increased_hunger * 12 + frequent_urination * 15 + excessive_thirst * 14 + unexplained_weight_loss * 10 + blurred_vision * 12, 70, 240))
    blood_pressure = int(_clamp(68 + age * 0.2 + fatigue_tiredness * 3 + blurred_vision * 2, 50, 130))
    skin_thickness = int(_clamp(18 + increased_hunger * 2 + fatigue_tiredness * 2 + frequent_urination, 7, 60))
    insulin = int(_clamp(50 + increased_hunger * 30 + excessive_thirst * 18 + frequent_urination * 22, 15, 320))
    bmi = round(_clamp(21 + increased_hunger * 1.4 + fatigue_tiredness * 0.8 - unexplained_weight_loss * 0.5, 16.0, 45.0), 1)
    diabetes_pedigree_function = round(_clamp(0.2 + (symptom_total / 18) * 1.1, 0.1, 2.5), 3)

    features = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age,
    }

    return {feature: features[feature] for feature in DIABETES_FEATURES}
