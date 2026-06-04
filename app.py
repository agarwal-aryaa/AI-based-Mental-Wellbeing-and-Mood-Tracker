import os, json, re, hashlib, pickle, random
from datetime import datetime
from collections import Counter
from flask import (Flask, render_template, request,redirect, url_for, session, jsonify)

app = Flask(__name__)
app.secret_key = "smartjournal_2024_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "users.json")
MODEL_PKL = os.path.join(BASE_DIR, "models", "mood_classifier.pkl")
REPORT_JSON = os.path.join(BASE_DIR, "models", "training_report.json")
CM_PNG = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
ACC_PNG = os.path.join(BASE_DIR, "models", "accuracy_comparison.png")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

MOOD_LABELS = ["happy", "neutral", "sad", "anxious", "angry"]

def load_users():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE) as f:
        return json.load(f)

def save_users(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_current_user():
    data = load_users()
    uid  = session.get("user_id")
    if not uid or uid not in data:
        session.clear()
        return None, None, None
    return data, uid, data[uid]

_model = None

def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, "rb") as f:
            _model = pickle.load(f)
    return _model

def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"&\w+;","", text)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()

def classify(text):
    mdl = get_model()
    if mdl is None:
        return "neutral", 0
    c = clean_text(text)
    if not c:
        return "neutral", 0
    try:
        pred  = mdl.predict([c])[0]
        proba = mdl.predict_proba([c])[0]
        return pred, int(max(proba) * 100)
    except Exception:
        return mdl.predict([c])[0], 0

def analyze_trends(records):
    if not records:
        return {}
    moods = [r["mood"] for r in records[-14:]]
    counts = Counter(moods)
    neg = sum(counts.get(m, 0) for m in ("sad","anxious","angry"))
    pos = counts.get("happy", 0)

    NEGATIVE = {"sad","anxious","angry"}
    best_streak, cur = 0, 0
    for m in moods:
        if m in NEGATIVE:
            cur += 1; best_streak = max(best_streak, cur)
        else:
            cur = 0
    return {
        "counts": dict(counts), "negative_ratio": round(neg / len(moods), 2),"positive_ratio": round(pos / len(moods), 2),"longest_neg_streak":  best_streak,"total": len(records),
    }

TIPS = {
    "happy": ["Keep a gratitude journal today","Share your positivity with someone","Channel this energy into a creative project"],
    "neutral": ["Try 5 minutes of deep breathing","Take a short mindful walk outside","Write down 3 things you appreciate today"],
    "sad": ["Be gentle with yourself — tough days pass","Reach out to a trusted friend or family member","Try light exercise like a 10-minute walk","Journal about what's weighing on you "],
    "anxious": ["Try the 4-7-8 breathing technique","Ground yourself: name 5 things you can see","Limit caffeine and screen time right now ","Write your worries down, then set them aside "],
    "angry": ["Take slow deep breaths before reacting ","Go for a brisk walk to release tension ","Give yourself space before addressing the situation"],
}

def get_recs(mood, trends):
    recs = TIPS.get(mood, TIPS["neutral"])[:]
    if trends.get("longest_neg_streak", 0) >= 3:
        recs.append("You've had several difficult days. Consider speaking to a mental health professional.")
    return recs

def check_alert(records, user):
    if len(records) < 3:
        return None
    last3 = [r["mood"] for r in records[-3:]]
    if all(m in ("sad","anxious","angry") for m in last3):
        return {"level":"warning","message":"You've logged negative emotions 3 days in a row. Please reach out to someone you trust.","guardian": user.get("guardian_email","")}
    if len(records) >= 7:
        neg7 = sum(1 for r in records[-7:] if r["mood"] in ("sad","anxious","angry"))
        if neg7 >= 6:
            return {"level":"critical","message":"Prolonged negative mood pattern detected. Please seek professional support immediately.","guardian": user.get("guardian_email","")}
    return None

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        data = load_users()
        uid  = request.form["username"].strip().lower()
        if not uid:
            return render_template("register.html", error="Username cannot be empty.")
        if uid in data:
            return render_template("register.html", error="Username already taken.")
        data[uid] = {"name": request.form["name"].strip(),"age": request.form.get("age",""),"password": hash_pw(request.form["password"]),"guardian_email":request.form.get("guardian_email","").strip(),"records": []}
        save_users(data)
        session["user_id"] = uid
        return redirect(url_for("dashboard"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        data = load_users()
        uid  = request.form["username"].strip().lower()
        pw   = hash_pw(request.form["password"])
        if uid in data and data[uid]["password"] == pw:
            session["user_id"] = uid
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    data, uid, user = get_current_user()
    if user is None:
        return redirect(url_for("login"))
    records = user["records"]
    trends = analyze_trends(records)
    alert = check_alert(records, user)
    last_mood = records[-1]["mood"] if records else "neutral"
    recs = get_recs(last_mood, trends)
    recent = list(reversed(records[-5:]))
    mood_map = {"happy":5,"neutral":3,"sad":1,"anxious":2,"angry":1}
    chart_labels = [r["date"] for r in records[-14:]]
    chart_values = [mood_map.get(r["mood"],3) for r in records[-14:]]
    mood_dist = trends.get("counts", {})
    return render_template("dashboard.html",user=user, recent=recent, trends=trends,alert=alert, recs=recs, last_mood=last_mood,chart_labels=json.dumps(chart_labels),chart_values=json.dumps(chart_values),mood_dist=json.dumps(mood_dist))

@app.route("/log", methods=["GET","POST"])
def log_mood():
    data, uid, user = get_current_user()
    if user is None:
        return redirect(url_for("login"))
    if request.method == "POST":
        mood  = request.form.get("mood","").strip()
        journal = request.form.get("journal","").strip()
        ai_used = False
        confidence = 0
        if journal:
            ml_mood, confidence = classify(journal)
            ai_used = True
            if not mood:
                mood = ml_mood
            elif confidence >= 60 and ml_mood != "neutral":
                mood = ml_mood
        mood = mood or "neutral"
        data[uid]["records"].append({"date": datetime.now().strftime("%Y-%m-%d"),"time": datetime.now().strftime("%H:%M"),"mood": mood,"journal": journal,"ai_used": ai_used,"confidence": confidence,
        })
        save_users(data)
        return redirect(url_for("dashboard"))
    model_loaded = get_model() is not None
    return render_template("log.html", model_loaded=model_loaded)

@app.route("/history")
def history():
    data, uid, user = get_current_user()
    if user is None:
        return redirect(url_for("login"))
    records = list(reversed(user["records"]))
    return render_template("history.html", user=user, records=records)

@app.route("/analysis")
def analysis():
    data, uid, user = get_current_user()
    if user is None:
        return redirect(url_for("login"))
    records = user["records"]
    trends  = analyze_trends(records)

    train_report = {}
    if os.path.exists(REPORT_JSON):
        with open(REPORT_JSON) as f:
            train_report = json.load(f)

    has_cm  = os.path.exists(CM_PNG)
    has_acc = os.path.exists(ACC_PNG)

    return render_template("analysis.html",
        user=user, trends=trends,
        all_records=json.dumps([{"date":r["date"],"mood":r["mood"]} for r in records]),
        train_report=train_report,
        has_cm=has_cm, has_acc=has_acc,
    )

@app.route("/models/<path:filename>")
def model_img(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(BASE_DIR,"models"), filename)

@app.route("/chatbot")
def chatbot():
    _, _, user = get_current_user()
    if user is None:
        return redirect(url_for("login"))
    return render_template("chatbot.html", user=user)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    _, _, user = get_current_user()
    if user is None:
        return jsonify({"error":"Unauthorized"}), 401
    msg = (request.json or {}).get("message","").strip()
    if not msg:
        return jsonify({"response":"Please say something."})
    mood, confidence = classify(msg)
    recs = get_recs(mood, analyze_trends(user["records"]))
    RESPONSES = {
        "happy": [f"That's wonderful, {user['name']}! It sounds like you're having a great time.","Your positivity is contagious! Keep nurturing those good vibes."],
        "sad": [f"I'm sorry you're feeling down, {user['name']}. It's okay to have tough days.", "You're not alone. Would you like to talk more about what's going on?"],
        "anxious": [f"I hear you, {user['name']}. Let's slow down together for a moment.", "Anxiety can feel overwhelming. Try taking 3 slow, deep breaths right now."],
        "angry": [f"It sounds like you're frustrated, {user['name']}. That's completely valid.","Sometimes we just need to vent. I'm here to listen without judgment."],
        "neutral": [f"Thanks for sharing, {user['name']}. How has your day been overall? ","I'm here whenever you want to talk."],
    }
    return jsonify({
        "response": random.choice(RESPONSES.get(mood, RESPONSES["neutral"])),
        "detected_mood": mood,
        "confidence": confidence,
        "tip": random.choice(recs),
    })

@app.route("/api/preview", methods=["POST"])
def api_preview():
    _, _, user = get_current_user()
    if user is None:
        return jsonify({"error":"Unauthorized"}), 401
    text = (request.json or {}).get("text","").strip()
    if len(text) < 8:
        return jsonify({"mood": None, "confidence": 0})
    mood, conf = classify(text)
    return jsonify({"mood": mood, "confidence": conf})

@app.route("/profile", methods=["GET","POST"])
def profile():
    data, uid, user = get_current_user()
    if user is None:
        return redirect(url_for("login"))
    if request.method == "POST":
        data[uid]["name"] = request.form.get("name", user["name"]).strip()
        data[uid]["age"] = request.form.get("age",  user.get("age",""))
        data[uid]["guardian_email"] = request.form.get("guardian_email", user.get("guardian_email","")).strip()
        save_users(data)
        return redirect(url_for("dashboard"))
    return render_template("profile.html", user=user, uid=uid)

if __name__ == "__main__":
    if not os.path.exists(MODEL_PKL):
        print("\nModel not found. Run  python train_model.py  first!\n")
    app.run(debug=True)
