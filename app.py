import os
import io
import json
import sqlite3
import uuid
from datetime import datetime
from functools import wraps

import torch
from PIL import Image
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    flash, abort, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PROCESSOR_DIR = os.path.join(MODEL_DIR, "processor")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_swin_model.pth")
DATABASE_PATH = os.path.join(BASE_DIR, "oralscan.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "user_uploads")
MAX_UPLOAD_MB = 8
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
# Set ORALSCAN_SECRET_KEY in your environment before deploying publicly.
app.secret_key = os.environ.get("ORALSCAN_SECRET_KEY", "change-this-local-development-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["CANCER", "NON CANCER"]

def init_db():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                cancer_probability REAL NOT NULL,
                non_cancer_probability REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.commit()

def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "info")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    if not os.path.isdir(PROCESSOR_DIR):
        raise FileNotFoundError(
            f"Processor folder missing: {PROCESSOR_DIR}. Add preprocessor_config.json there."
        )
    processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint missing: {CHECKPOINT_PATH}. Copy best_swin_model.pth into model/."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    # Your Colab checkpoint was saved as a state_dict OrderedDict.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()
    return processor, model

init_db()
processor, model = load_model()

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
    predicted_index = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": float(probs[predicted_index] * 100),
        "cancer_probability": float(probs[0] * 100),
        "non_cancer_probability": float(probs[1] * 100),
    }

@app.context_processor
def inject_user():
    return {"current_username": session.get("username")}

@app.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if len(username) < 3 or len(username) > 32:
            flash("Username must be between 3 and 32 characters.", "error")
        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        else:
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                        (username, generate_password_hash(password), datetime.now().isoformat(timespec="seconds"))
                    )
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username is already taken.", "error")
    return render_template("auth.html", mode="signup")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ? COLLATE NOCASE",
                (username,)
            ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        flash("Incorrect username or password.", "error")
    return render_template("auth.html", mode="login")

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    uploaded = request.files.get("image")
    if not uploaded or uploaded.filename == "":
        flash("Choose an image or capture one with your camera first.", "error")
        return redirect(url_for("home"))
    if not allowed_file(uploaded.filename):
        flash("Please upload a JPG, JPEG, PNG, or WEBP image.", "error")
        return redirect(url_for("home"))
    image_bytes = uploaded.read()
    try:
        # Validate that this is a readable image before saving it.
        Image.open(io.BytesIO(image_bytes)).verify()
        result = predict_image(image_bytes)
    except Exception:
        flash("We couldn't read that image. Please try another image.", "error")
        return redirect(url_for("home"))

    safe_original = secure_filename(uploaded.filename) or "uploaded-image"
    ext = safe_original.rsplit(".", 1)[1].lower()
    stored_filename = f"user{session['user_id']}_{uuid.uuid4().hex}.{ext}"
    user_folder = os.path.join(UPLOAD_FOLDER, str(session["user_id"]))
    os.makedirs(user_folder, exist_ok=True)
    with open(os.path.join(user_folder, stored_filename), "wb") as f:
        f.write(image_bytes)

    created_at = datetime.now().isoformat(timespec="seconds")
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions
            (user_id, image_filename, original_filename, prediction, confidence,
             cancer_probability, non_cancer_probability, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"], stored_filename, safe_original, result["prediction"],
            result["confidence"], result["cancer_probability"],
            result["non_cancer_probability"], created_at
        ))
        prediction_id = cursor.lastrowid
    return redirect(url_for("prediction_detail", prediction_id=prediction_id))

@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        records = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
            (session["user_id"],)
        ).fetchall()
    return render_template("history.html", records=records)

@app.route("/prediction/<int:prediction_id>")
@login_required
def prediction_detail(prediction_id):
    with get_db() as conn:
        record = conn.execute(
            "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, session["user_id"])
        ).fetchone()
    if record is None:
        abort(404)
    return render_template("detail.html", record=record)

@app.route("/images/<int:prediction_id>")
@login_required
def prediction_image(prediction_id):
    with get_db() as conn:
        record = conn.execute(
            "SELECT image_filename FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, session["user_id"])
        ).fetchone()
    if record is None:
        abort(404)
    user_folder = os.path.join(UPLOAD_FOLDER, str(session["user_id"]))
    return send_from_directory(user_folder, record["image_filename"], as_attachment=False)

@app.route("/prediction/<int:prediction_id>/delete", methods=["POST"])
@login_required
def delete_prediction(prediction_id):
    with get_db() as conn:
        record = conn.execute(
            "SELECT image_filename FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, session["user_id"])
        ).fetchone()
        if record is None:
            abort(404)
        image_path = os.path.join(UPLOAD_FOLDER, str(session["user_id"]), record["image_filename"])
        conn.execute("DELETE FROM predictions WHERE id = ? AND user_id = ?",
                     (prediction_id, session["user_id"]))
    if os.path.isfile(image_path):
        os.remove(image_path)
    flash("Prediction and its image were deleted.", "success")
    return redirect(url_for("history"))

@app.errorhandler(413)
def too_large(_error):
    flash(f"Image is too large. Maximum size is {MAX_UPLOAD_MB} MB.", "error")
    return redirect(url_for("home"))

@app.errorhandler(404)
def not_found(_error):
    return render_template("not_found.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
