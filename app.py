import os
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from database.auth_db import (
    AuthError,
    authenticate_user,
    create_user,
    get_user_auth_record,
    get_user_by_id,
    init_auth_table,
)
from database.db import fetch_prediction_history, init_db, save_prediction
from services.prediction_service import (
    PredictionServiceError,
    predict_brain_tumor,
    predict_diabetes,
    predict_heart,
)
from training.config import PROJECT_DISCLAIMER


PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "disease-project-dev-secret")
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

init_db()
init_auth_table()


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if session.get("user_id"):
            return view_function(*args, **kwargs)

        if request.path.startswith("/api/"):
            return jsonify({"error": "Please log in first."}), 401

        return redirect(url_for("login"))

    return wrapped_view


@app.get("/")
def home():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.get("/login")
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("login.html", disclaimer=PROJECT_DISCLAIMER)


@app.post("/login")
def login_post():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    user_record = get_user_auth_record(username)
    user = authenticate_user(username, password)

    if user_record is None:
        session.clear()
        flash("Account not found. Please sign up first before logging in.", "error")
        return redirect(url_for("login"))

    if user is None:
        session.clear()
        flash("Incorrect username or password. Please try again.", "error")
        return redirect(url_for("login"))

    session["user_id"] = user["id"]
    flash("Login successful.", "success")
    return redirect(url_for("dashboard"))


@app.get("/register")
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("register.html", disclaimer=PROJECT_DISCLAIMER)


@app.post("/register")
def register_post():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    confirm_password = request.form.get("confirm_password") or ""

    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect(url_for("register"))

    try:
        user = create_user(username, password)
    except AuthError as error:
        flash(str(error), "error")
        return redirect(url_for("register"))

    session["user_id"] = user["id"]
    flash("Account created successfully.", "success")
    return redirect(url_for("dashboard"))


@app.get("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.get("/dashboard")
@login_required
def dashboard():
    return render_template("index.html", user=current_user(), disclaimer=PROJECT_DISCLAIMER)


@app.post("/api/predict/heart")
@login_required
def heart_prediction():
    payload = request.get_json(force=True, silent=True) or {}

    try:
        result = predict_heart(payload)
        save_prediction(session["user_id"], "heart", payload, result)
        return jsonify(result)
    except PredictionServiceError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/predict/diabetes")
@login_required
def diabetes_prediction():
    payload = request.get_json(force=True, silent=True) or {}

    try:
        result = predict_diabetes(payload)
        save_prediction(session["user_id"], "diabetes", payload, result)
        return jsonify(result)
    except PredictionServiceError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/predict/brain-tumor")
@login_required
def brain_tumor_prediction():
    uploaded_file = request.files.get("scan")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload a brain MRI scan image."}), 400

    try:
        result = predict_brain_tumor(uploaded_file.stream, uploaded_file.filename)
        save_prediction(session["user_id"], "brain_tumor", {"filename": uploaded_file.filename}, result)
        return jsonify(result)
    except PredictionServiceError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500


@app.get("/api/predictions/history")
@login_required
def prediction_history():
    return jsonify({"predictions": fetch_prediction_history(session["user_id"], limit=30)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
