"""Flask web application: login, upload, prediction, visualisation, dashboard and history."""
import io
import json
import os
import secrets
import sqlite3
from datetime import datetime
from functools import wraps

import joblib
import pandas as pd
from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request,
                   send_file, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from preprocess import CLASSES, FEATURES, load_csv, map_label

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
UPLOAD_DIR = os.path.join(BASE, "uploads")
INSTANCE_DIR = os.path.join(BASE, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "nids.db")
ALLOWED_EXT = {".csv", ".txt"}
MAX_STORED_ROWS = 20000
PAGE_SIZE = 50
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)


def _secret_key():
    env = os.environ.get("NIDS_SECRET")
    if env:
        return env
    path = os.path.join(INSTANCE_DIR, "secret.key")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(secrets.token_hex(32))
    with open(path) as fh:
        return fh.read().strip()


app = Flask(__name__)
app.config.update(SECRET_KEY=_secret_key(), MAX_CONTENT_LENGTH=50 * 1024 * 1024,
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            total INTEGER NOT NULL,
            normal_count INTEGER NOT NULL,
            attack_count INTEGER NOT NULL,
            class_counts TEXT NOT NULL,
            accuracy REAL,
            created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
            row_no INTEGER NOT NULL,
            protocol TEXT, service TEXT, src_bytes REAL,
            actual TEXT,
            predicted TEXT NOT NULL,
            confidence REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_pred_det ON predictions(detection_id, predicted);
    """)
    if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        db.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
                   ("admin", generate_password_hash("admin123"), now()))
        print("Created default account  ->  username: admin   password: admin123")
    db.commit()
    db.close()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- auth + csrf
def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*a, **kw)
    return wrapped


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


@app.before_request
def check_csrf():
    if request.method == "POST":
        sent = request.form.get("csrf_token", "")
        if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
            abort(400, "Your session expired. Go back, reload the page and try again.")


app.jinja_env.globals.update(csrf_token=csrf_token, CLASSES=CLASSES)


@app.context_processor
def inject_user():
    return {"current_user": session.get("username")}


@app.errorhandler(413)
def too_large(_e):
    flash("That file is larger than 50 MB. Upload a smaller file.", "error")
    return redirect(url_for("predict"))


@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", message=getattr(e, "description", "Bad request")), 400


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        user = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear()
            session.update(user_id=user["id"], username=user["username"])
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
        flash("Username or password is incorrect.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not (3 <= len(username) <= 30) or not username.replace("_", "").replace(".", "").isalnum():
            flash("Username must be 3-30 characters: letters, numbers, dots or underscores.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != request.form.get("confirm", ""):
            flash("The two passwords do not match.", "error")
        else:
            try:
                db = get_db()
                db.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
                           (username, generate_password_hash(password), now()))
                db.commit()
                flash("Account created. Sign in to continue.", "ok")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username is taken. Choose another.", "error")
    return render_template("register.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------- model
_cache = {}


def get_model():
    """Load the trained model once. Returns (model, preprocessor) or (None, None) if not trained."""
    if "model" not in _cache:
        mp = os.path.join(MODEL_DIR, "nids_cnn.keras")
        pp = os.path.join(MODEL_DIR, "preprocessor.joblib")
        if not (os.path.exists(mp) and os.path.exists(pp)):
            return None, None
        from tensorflow import keras
        _cache["model"] = keras.models.load_model(mp)
        _cache["pre"] = joblib.load(pp)
    return _cache["model"], _cache["pre"]


def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------- pages
def user_stats(user_id):
    db = get_db()
    row = db.execute("""SELECT COUNT(*) scans, COALESCE(SUM(total),0) records,
                        COALESCE(SUM(attack_count),0) attacks FROM detections WHERE user_id=?""",
                     (user_id,)).fetchone()
    totals = {c: 0 for c in CLASSES}
    for r in db.execute("SELECT class_counts FROM detections WHERE user_id=?", (user_id,)):
        for k, v in json.loads(r["class_counts"]).items():
            totals[k] = totals.get(k, 0) + v
    recent = db.execute("""SELECT id, filename, total, attack_count, created_at FROM detections
                           WHERE user_id=? ORDER BY id DESC LIMIT 10""", (user_id,)).fetchall()
    return row, totals, list(reversed(recent))


@app.route("/")
@login_required
def dashboard():
    row, totals, recent = user_stats(session["user_id"])
    latest = get_db().execute("SELECT * FROM detections WHERE user_id=? ORDER BY id DESC LIMIT 1",
                              (session["user_id"],)).fetchone()
    latest_counts = json.loads(latest["class_counts"]) if latest else None
    return render_template(
        "dashboard.html", stats=row, totals=totals, latest=latest, latest_counts=latest_counts,
        recent=list(reversed(recent)), trained=os.path.exists(os.path.join(MODEL_DIR, "nids_cnn.keras")),
        rate=(row["attacks"] / row["records"] * 100) if row["records"] else 0,
        chart={"totals": totals,
               "scans": [{"name": r["filename"], "normal": r["total"] - r["attack_count"],
                          "attack": r["attack_count"]} for r in recent]})


@app.route("/api/stats")
@login_required
def api_stats():
    row, totals, recent = user_stats(session["user_id"])
    return jsonify(scans=row["scans"], records=row["records"], attacks=row["attacks"],
                   class_totals=totals,
                   recent=[dict(r) for r in recent])


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    trained = os.path.exists(os.path.join(MODEL_DIR, "nids_cnn.keras"))
    if request.method == "GET":
        return render_template("predict.html", trained=trained)

    f = request.files.get("dataset")
    if not f or not f.filename:
        flash("Choose a CSV file to analyse.", "error")
        return redirect(url_for("predict"))
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        flash("Only .csv or .txt files are supported.", "error")
        return redirect(url_for("predict"))
    model, pre = get_model()
    if model is None:
        flash("No trained model found. Run python train_model.py first.", "error")
        return redirect(url_for("predict"))

    safe = secure_filename(f.filename) or "upload.csv"
    path = os.path.join(UPLOAD_DIR, f"{secrets.token_hex(6)}_{safe}")
    f.save(path)
    try:
        df = load_csv(path)
        if df.empty:
            raise ValueError("The file has no rows.")
        X = pre.transform(df)
        probs = model.predict(X, batch_size=1024, verbose=0)
    except Exception as exc:  # report a friendly message for malformed files
        flash(f"Could not analyse that file: {exc}", "error")
        return redirect(url_for("predict"))
    finally:
        if os.path.exists(path):
            os.remove(path)

    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    predicted = [CLASSES[i] for i in idx]
    actual = df["label"].map(map_label) if "label" in df.columns else None
    accuracy = None
    if actual is not None and actual.notna().any():
        known = actual.notna().values
        accuracy = float((pd.Series(predicted)[known].values == actual[known].values).mean())

    counts = {c: predicted.count(c) for c in CLASSES}
    attacks = len(predicted) - counts["normal"]
    db = get_db()
    cur = db.execute("""INSERT INTO detections (user_id, filename, total, normal_count, attack_count,
                        class_counts, accuracy, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                     (session["user_id"], f.filename, len(predicted), counts["normal"], attacks,
                      json.dumps(counts), accuracy, now()))
    det_id = cur.lastrowid
    n = min(len(predicted), MAX_STORED_ROWS)
    db.executemany(
        "INSERT INTO predictions (detection_id,row_no,protocol,service,src_bytes,actual,predicted,confidence) VALUES (?,?,?,?,?,?,?,?)",
        [(det_id, i + 1, str(df["protocol_type"].iat[i]), str(df["service"].iat[i]),
          float(pd.to_numeric(df["src_bytes"].iat[i], errors="coerce") or 0),
          (None if actual is None or pd.isna(actual.iat[i]) else actual.iat[i]),
          predicted[i], float(conf[i])) for i in range(n)])
    db.commit()
    return redirect(url_for("results", det_id=det_id))


def _own_detection(det_id):
    row = get_db().execute("SELECT * FROM detections WHERE id=? AND user_id=?",
                           (det_id, session["user_id"])).fetchone()
    if row is None:
        abort(404)
    return row


@app.route("/results/<int:det_id>")
@login_required
def results(det_id):
    det = _own_detection(det_id)
    show = request.args.get("show", "attacks")
    page = max(request.args.get("page", 1, type=int), 1)
    where, args = "detection_id=?", [det_id]
    if show == "attacks":
        where += " AND predicted != 'normal'"
    elif show in CLASSES:
        where += " AND predicted=?"
        args.append(show)
    db = get_db()
    total_rows = db.execute(f"SELECT COUNT(*) FROM predictions WHERE {where}", args).fetchone()[0]
    rows = db.execute(f"SELECT * FROM predictions WHERE {where} ORDER BY row_no LIMIT ? OFFSET ?",
                      args + [PAGE_SIZE, (page - 1) * PAGE_SIZE]).fetchall()
    return render_template("results.html", det=det, counts=json.loads(det["class_counts"]), rows=rows,
                           show=show, page=page, pages=max((total_rows + PAGE_SIZE - 1) // PAGE_SIZE, 1),
                           total_rows=total_rows, truncated=det["total"] > MAX_STORED_ROWS)


@app.route("/results/<int:det_id>/download")
@login_required
def download(det_id):
    det = _own_detection(det_id)
    rows = get_db().execute("""SELECT row_no, protocol, service, src_bytes, actual, predicted, confidence
                               FROM predictions WHERE detection_id=? ORDER BY row_no""", (det_id,)).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    buf = io.BytesIO(df.to_csv(index=False).encode())
    name = os.path.splitext(secure_filename(det["filename"]) or "results")[0]
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"{name}_predictions.csv")


@app.route("/history")
@login_required
def history():
    rows = get_db().execute("SELECT * FROM detections WHERE user_id=? ORDER BY id DESC",
                            (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/history/<int:det_id>/delete", methods=["POST"])
@login_required
def delete_detection(det_id):
    _own_detection(det_id)
    db = get_db()
    db.execute("DELETE FROM detections WHERE id=?", (det_id,))
    db.commit()
    flash("Scan deleted.", "ok")
    return redirect(url_for("history"))


@app.route("/model")
@login_required
def model_page():
    return render_template("model.html", m=load_metrics())


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
else:
    init_db()
