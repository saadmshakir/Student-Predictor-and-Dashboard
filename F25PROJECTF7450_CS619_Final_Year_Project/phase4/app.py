# CS619 — Student Performance Predictor
# Phase 4 FINAL — MySQL 8.0 + Flask + Recommendations Engine
# Authors: Syeda Farkhanda Batool Naqvi (BC220424824)
#          Saad Shakir (BC220404989)

from flask import Flask, request, redirect, url_for, send_file, session, render_template
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
import io, os, warnings
warnings.filterwarnings("ignore")

# MySQL 
try:
    import pymysql
    import pymysql.cursors
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("WARNING: pymysql not installed.  Run:  pip install pymysql cryptography")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "cs619_phase4_dev_key")

#  DB Config 
#  TO SET DB PASSWORD (run ONCE in CMD before starting app.py):
#   Windows CMD:        set DB_PASSWORD=our_mysql_root_password
#   Windows PowerShell: $env:DB_PASSWORD="our_mysql_root_password"
DB_CONFIG = {
    "host":        "localhost",
    "user":        "root",
    "password":    os.environ.get("DB_PASSWORD", ""),
    "database":    "cs619_studentdb",
    "charset":     "utf8mb4",
    "autocommit":  True,
    "ssl_disabled": True,
}
if MYSQL_AVAILABLE:
    DB_CONFIG["cursorclass"] = pymysql.cursors.DictCursor

if not os.environ.get("DB_PASSWORD"):
    print("=" * 60)
    print("  WARNING: DB_PASSWORD not set!")
    print("  Run:  set DB_PASSWORD=our_mysql_root_password")
    print("=" * 60)

def get_conn():
    if not MYSQL_AVAILABLE:
        raise RuntimeError("pymysql not installed.  pip install pymysql cryptography")
    return pymysql.connect(**DB_CONFIG)

#  Numeric columns to coerce 
_NUM_COLS = [
    "Age","Semester","Total_Classes","Classes_Attended","Attendance_Percentage",
    "Assignment_1","Assignment_2","Assignment_3","Average_Assignment",
    "Quiz_1","Quiz_2","Average_Quiz","Midterm_Exam","Final_Exam",
    "Lab_Work_1","Lab_Work_2","Average_Lab","Project_Score","CGPA","Previous_GPA",
    "Study_Hours_Per_Week","Library_Visits","Online_Course_Hours",
    "Class_Participation","Group_Work_Score","Total_Score"
]

def _fix_types(df):
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)

#  Data Loading (MySQL with CSV fallback)
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "processed_student_data.csv")
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM students")
            count = cur.fetchone()["cnt"]

        if count == 0:
            print("First run — importing CSV into MySQL...")
            df_csv = pd.read_csv(csv_path)
            df_csv = _fix_types(df_csv)
            # Only keep columns that exist in the DB schema
            db_cols = [
                "Student_ID","Registration_No","Gender","Age","Course","Semester","Section",
                "Total_Classes","Classes_Attended","Attendance_Percentage",
                "Assignment_1","Assignment_2","Assignment_3","Average_Assignment",
                "Quiz_1","Quiz_2","Average_Quiz","Midterm_Exam","Final_Exam",
                "Lab_Work_1","Lab_Work_2","Average_Lab","Project_Score","CGPA",
                "Previous_GPA","Study_Hours_Per_Week","Library_Visits",
                "Online_Course_Hours","Class_Participation","Group_Work_Score",
                "Sports_Activity","Club_Membership","Total_Score","Performance"
            ]
            cols = [c for c in db_cols if c in df_csv.columns]
            ph   = ", ".join(["%s"] * len(cols))
            cs   = ", ".join([f"`{c}`" for c in cols])
            sql  = f"INSERT IGNORE INTO students ({cs}) VALUES ({ph})"
            with conn.cursor() as cur:
                for _, row in df_csv[cols].iterrows():
                    cur.execute(sql, tuple(row))
            conn.commit()
            conn.close()
            print(f"SUCCESS: {len(df_csv)} students imported!")
            return df_csv
        else:
            print(f"Loading {count} students from MySQL...")
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM students")
                rows = cur.fetchall()
            conn.close()
            df = pd.DataFrame(rows)
            if "id" in df.columns:
                df = df.drop(columns=["id"])
            df = _fix_types(df)
            print(f"MySQL load OK: {df.shape}")
            return df
    except Exception as e:
        print(f"MySQL error: {e}")
        print("Falling back to CSV...")
        return _fix_types(pd.read_csv(csv_path))

def save_predictions(df):
    if not MYSQL_AVAILABLE:
        return
    try:
        conn = get_conn()
        sql = """INSERT INTO predictions
                    (Student_ID, Predicted_Result, Fail_Probability, Risk_Level)
                 VALUES (%s,%s,%s,%s)
                 ON DUPLICATE KEY UPDATE
                    Predicted_Result=VALUES(Predicted_Result),
                    Fail_Probability=VALUES(Fail_Probability),
                    Risk_Level=VALUES(Risk_Level)"""
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                fp = float(row.get("Fail_Probability", 0))
                if   fp >= 70: risk = "Critical"
                elif fp >= 50: risk = "High"
                elif fp >= 30: risk = "Medium"
                else:          risk = "Low"
                cur.execute(sql, (row["Student_ID"],
                                  row.get("Final_Result", "N/A"), fp, risk))
        conn.commit()
        conn.close()
        print("Predictions saved to MySQL.")
    except Exception as e:
        print(f"Could not save predictions: {e}")

# Load ML Artifacts (Phase 3 pkl files) 
BASE = os.path.dirname(os.path.abspath(__file__))

try:
    model          = joblib.load(os.path.join(BASE, "student_model_v1.pkl"))
    scaler         = joblib.load(os.path.join(BASE, "student_scaler_v1.pkl"))
    target_encoder = joblib.load(os.path.join(BASE, "student_encoder_v1.pkl"))
    feat_encoders  = joblib.load(os.path.join(BASE, "student_feature_encoders_v1.pkl"))
    print(f"ML artifacts loaded: {type(model).__name__}")
except FileNotFoundError as e:
    print(f"ERROR loading model: {e}")
    print("Make sure pkl files are in the same folder as app.py")
    exit(1)

#  Feature Configuration (matches Phase 3 training pipeline) 
# 26 numeric features — scaled by StandardScaler
NUMERIC_FEATURES = [
    "Age","Semester","Total_Classes","Classes_Attended","Attendance_Percentage",
    "Assignment_1","Assignment_2","Assignment_3","Average_Assignment",
    "Quiz_1","Quiz_2","Average_Quiz","Midterm_Exam","Final_Exam",
    "Lab_Work_1","Lab_Work_2","Average_Lab","Project_Score","CGPA","Previous_GPA",
    "Study_Hours_Per_Week","Library_Visits","Online_Course_Hours",
    "Class_Participation","Group_Work_Score","Total_Score"
]
# 5 categorical features — label-encoded then appended (not scaled)
CATEGORICAL_FEATURES = ["Gender","Course","Section","Sports_Activity","Club_Membership"]

# At-Risk class index in LabelEncoder
AT_RISK_IDX = list(target_encoder.classes_).index("At-Risk")

#  Prediction Pipeline 

def predict_all_students(data):
    """
    Bulletproof pipeline — identical to Phase 3, immune to MySQL extra columns.

    Root cause of the '32 features, expected 31' error:
      MySQL SELECT * returns an 'id' auto-increment column (or other extra cols)
      that survive into X. Dropping by name is fragile.
      FIX: explicitly select EXACTLY the 31 columns the model was trained on.

    These 31 columns are derived from Phase 2:
      CSV (34 cols) minus Student_ID, Registration_No, Performance = 31 features
      in this exact order (same as Phase 2 training dataframe column order).
    """
    # The 31  features Phase 2 trained on 
    MODEL_COLS = [
        "Gender", "Age", "Course", "Semester", "Section",
        "Total_Classes", "Classes_Attended", "Attendance_Percentage",
        "Assignment_1", "Assignment_2", "Assignment_3", "Average_Assignment",
        "Quiz_1", "Quiz_2", "Average_Quiz",
        "Midterm_Exam", "Final_Exam",
        "Lab_Work_1", "Lab_Work_2", "Average_Lab",
        "Project_Score", "CGPA", "Previous_GPA",
        "Study_Hours_Per_Week", "Library_Visits", "Online_Course_Hours",
        "Class_Participation", "Group_Work_Score",
        "Sports_Activity", "Club_Membership",
        "Total_Score",
    ]

    df = data.copy()

    # Step A — encode categoricals in-place before selecting MODEL_COLS
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and col in feat_encoders:
            le    = feat_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).map(
                lambda x, k=known, c=le.classes_: x if x in k else c[0]
            )
            df[col] = le.transform(df[col])
        else:
            df[col] = 0  # column missing entirely — fill with 0

    # Step B — EXPLICITLY select only the 31 model features (ignores id, extras)
    for col in MODEL_COLS:
        if col not in df.columns:
            df[col] = 0  # safety: add missing col as 0
    X = df[MODEL_COLS].copy()  # guaranteed exactly 31 columns, correct order

    # Step C — coerce to numeric, triple-layer NaN/inf cleanup
    for col in MODEL_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    col_medians = X.median()          # compute before any fill
    X = X.fillna(col_medians)         # fill NaN with column median
    X = X.fillna(0)                   # fill if median itself was NaN
    X = X.replace([np.inf, -np.inf], 0)

    # Step D — scale numeric features in-place (same 26 cols Phase 2 used)
    num_present = [c for c in NUMERIC_FEATURES if c in X.columns]
    X[num_present] = scaler.transform(X[num_present])
    X = X.fillna(0).replace([np.inf, -np.inf], 0)  # post-scale safety

    # Step E — predict (X is now exactly 31 cols, correct order)
    preds_enc = model.predict(X.values)
    proba     = model.predict_proba(X.values)

    # Step F — decode; Average = Pass, only At-Risk = Fail
    performance = target_encoder.inverse_transform(preds_enc)
    data["Performance"]      = performance
    PASS_CATS = {"Excellent", "Very Good", "Good", "Average"}
    data["Final_Result"]     = ["Pass" if p in PASS_CATS else "Fail"
                                for p in performance]
    data["Fail_Probability"] = (proba[:, AT_RISK_IDX] * 100).round(1)

    float_cols = data.select_dtypes(include=["float64", "float32"]).columns
    data[float_cols] = data[float_cols].round(2)
    return data

#  Startup 
df_full        = load_data()
df_final       = predict_all_students(df_full.copy())
total_students = len(df_final)
pass_count     = int((df_final["Final_Result"] == "Pass").sum())
fail_count     = int((df_final["Final_Result"] == "Fail").sum())
print("=" * 58)
print("   CS619 — Student Performance Prediction System")
print(f"   Phase 4 Dashboard  |  ML Model: {type(model).__name__}")
print("=" * 58)
print(f"   Students loaded   : {total_students:,}")
print(f"   Predicted Pass    : {pass_count:,}  ({round(pass_count/total_students*100,1)}%)")
print(f"   Predicted At-Risk : {fail_count:,}  ({round(fail_count/total_students*100,1)}%)")
print("=" * 58)
save_predictions(df_final)

#  Class Averages (computed once at startup) 
CLASS_AVG = {
    "Attendance_Percentage": round(df_final["Attendance_Percentage"].mean(), 1),
    "Average_Assignment":    round(df_final["Average_Assignment"].mean(), 1),
    "Average_Quiz":          round(df_final["Average_Quiz"].mean(), 1),
    "Midterm_Exam":          round(df_final["Midterm_Exam"].mean(), 1),
    "Final_Exam":            round(df_final["Final_Exam"].mean(), 1),
    "Average_Lab":           round(df_final["Average_Lab"].mean(), 1),
    "Project_Score":         round(df_final["Project_Score"].mean(), 1),
    "Study_Hours_Per_Week":  round(df_final["Study_Hours_Per_Week"].mean(), 1),
    "CGPA":                  round(df_final["CGPA"].mean(), 2),
    "Total_Score":           round(df_final["Total_Score"].mean(), 1),
}

# PHASE 4 — RECOMMENDATIONS ENGINE
# Analyzes 10 academic dimensions as required by Phase 4 :
#   1. Attendance            2. Average Assignment   3. Average Quiz
#   4. Midterm Exam          5. Final Exam            6. Study Hours
#   7. CGPA                  8. Lab Work              9. Project Score
#   10.  Extra Classes alert for at-risk students

def generate_recommendations(s):
    """
    Takes a single student's pandas Series and returns:
      recs        — list of recommendation card dicts
      alerts      — list of early alert strings (critical conditions)
      strengths   — list of strength badge strings
      risk_level  — "Critical" / "High" / "Medium" / "Low"
      risk_color  — CSS hex color
      risk_bg     — CSS hex background
    """
    recs, alerts, strengths = [], [], []
    fail_prob = float(s.get("Fail_Probability", 0))

    # Risk tier
    if   fail_prob >= 70: rl, rc, rb = "Critical", "#fc8181", "#fff5f5"
    elif fail_prob >= 50: rl, rc, rb = "High",     "#f6ad55", "#fffaf0"
    elif fail_prob >= 30: rl, rc, rb = "Medium",   "#63b3ed", "#ebf8ff"
    else:                 rl, rc, rb = "Low",       "#48bb78", "#f0fff4"

    #  1. Attendance 
    att    = float(s.get("Attendance_Percentage", 100))
    ca_att = CLASS_AVG["Attendance_Percentage"]
    if att < 60:
        alerts.append(
            f"⚠ CRITICAL: Attendance is {att}% — below the 60% eligibility threshold. "
            "You may be barred from sitting the final exam. Contact your course coordinator immediately."
        )
        recs.append({
            "icon":   "📅",
            "area":   "Attendance — Critical (below 60%)",
            "detail": (f"Your attendance: {att}% vs class average: {ca_att}%. "
                       "Students below 60% risk losing exam eligibility."),
            "action": ("Attend every remaining class without exception. "
                       "Speak to your academic advisor today to explore a recovery plan.")
        })
    elif att < 75:
        recs.append({
            "icon":   "📅",
            "area":   "Attendance — Below Required (75%)",
            "detail": (f"Your attendance: {att}% vs class average: {ca_att}%. "
                       "75% is the minimum required attendance threshold."),
            "action": ("Target 85%+ going forward. Set phone reminders for every lecture. "
                       "Missing one class = missing marks + content gaps.")
        })
    elif att >= 90:
        strengths.append(f"Excellent attendance: {att}%")

    # 2. Assignment Performance 
    asn    = float(s.get("Average_Assignment", 100))
    ca_asn = CLASS_AVG["Average_Assignment"]
    if asn < 40:
        alerts.append(
            f"Assignment average is critically low ({asn}/100). "
            "Assignments contribute directly to your final grade."
        )
        recs.append({
            "icon":   "📝",
            "area":   "Assignment Score — Critical (below 40)",
            "detail": (f"Your avg: {asn} vs class average: {ca_asn}. "
                       "Assignments account for a significant portion of the final grade."),
            "action": ("Submit all pending assignments immediately — partial marks are better than zero. "
                       "Visit office hours or join a study group to close knowledge gaps.")
        })
    elif asn < ca_asn - 10:
        recs.append({
            "icon":   "📝",
            "area":   "Assignment Score — Below Class Average",
            "detail": f"Your avg: {asn} vs class average: {ca_asn}.",
            "action": ("Review instructor feedback on graded assignments. "
                       "Reattempt questions you got wrong as self-practice before next submission.")
        })
    elif asn >= 80:
        strengths.append(f"Strong assignments: {asn}/100")

    # 3. Quiz Performance 
    qz    = float(s.get("Average_Quiz", 100))
    ca_qz = CLASS_AVG["Average_Quiz"]
    if qz < 40:
        recs.append({
            "icon":   "🧠",
            "area":   "Quiz Score — Serious Improvement Needed (below 40)",
            "detail": (f"Your quiz avg: {qz} vs class average: {ca_qz}. "
                       "Quizzes reflect weekly lecture comprehension."),
            "action": ("Review lecture slides the same day they are taught. "
                       "Use flashcards and past quiz papers to revise for each upcoming quiz.")
        })
    elif qz < ca_qz - 10:
        recs.append({
            "icon":   "🧠",
            "area":   "Quiz Score — Below Class Average",
            "detail": f"Your quiz avg: {qz} vs class average: {ca_qz}.",
            "action": ("Attend all lectures. Form a weekly quiz-prep group with 2–3 classmates "
                       "to review topics before each quiz.")
        })
    elif qz >= 80:
        strengths.append(f"Good quiz scores: {qz}/100")

    #  4. Midterm Exam 
    mid    = float(s.get("Midterm_Exam", 100))
    ca_mid = CLASS_AVG["Midterm_Exam"]
    if mid < 40:
        alerts.append(
            f"Midterm Exam score is critically low ({mid}/100). "
            "Midterm topics frequently reappear in the Final Exam — focused revision is essential."
        )
        recs.append({
            "icon":   "📋",
            "area":   "Midterm Exam — Critical (below 40)",
            "detail": (f"Your midterm: {mid} vs class average: {ca_mid}. "
                       "A low midterm substantially raises your fail probability."),
            "action": ("Request detailed feedback from your instructor immediately. "
                       "Spend extra time on midterm chapters — they are high-weight in finals.")
        })
    elif mid < ca_mid - 10:
        recs.append({
            "icon":   "📋",
            "area":   "Midterm Exam — Below Class Average",
            "detail": f"Your midterm: {mid} vs class average: {ca_mid}.",
            "action": ("Review your midterm paper with your instructor. "
                       "Identify weakest chapters and schedule focused revision sessions.")
        })
    elif mid >= 80:
        strengths.append(f"Strong midterm: {mid}/100")

    #  5. Final Exam 
    fe    = float(s.get("Final_Exam", 100))
    ca_fe = CLASS_AVG["Final_Exam"]
    if fe < 40:
        alerts.append(
            f"Final Exam score is critically low ({fe}/100). "
            "This is the single highest-weight component of your grade."
        )
        recs.append({
            "icon":   "🎯",
            "area":   "Final Exam — Critical (below 40)",
            "detail": (f"Your final exam: {fe} vs class average: {ca_fe}. "
                       "Final exam typically carries 40–50% of the total grade."),
            "action": ("Enroll in remedial/extra exam-prep sessions immediately. "
                       "Solve past papers under timed conditions daily. "
                       "Focus on high-mark question types from previous exams.")
        })
    elif fe < ca_fe - 10:
        recs.append({
            "icon":   "🎯",
            "area":   "Final Exam — Below Class Average",
            "detail": f"Your final exam: {fe} vs class average: {ca_fe}.",
            "action": ("Practice past final exam papers. "
                       "Prioritize topics with highest marks allocation. "
                       "Seek help from your instructor on areas you find difficult.")
        })
    elif fe >= 80:
        strengths.append(f"Excellent final exam: {fe}/100")

    #  6. Study Hours 
    sh    = float(s.get("Study_Hours_Per_Week", 100))
    ca_sh = CLASS_AVG["Study_Hours_Per_Week"]
    if sh < 5:
        recs.append({
            "icon":   "⏰",
            "area":   "Study Hours — Very Low (below 5 hrs/week)",
            "detail": (f"You study {sh} hrs/week vs class average: {ca_sh} hrs. "
                       "Insufficient study time is one of the strongest predictors of failure."),
            "action": ("Build a daily study schedule: 1–2 hours of focused revision per day. "
                       "Block distractions using apps like Forest or Cold Turkey. "
                       "Study in short 25-min Pomodoro blocks for better retention.")
        })
    elif sh < ca_sh - 5:
        recs.append({
            "icon":   "⏰",
            "area":   "Study Hours — Below Class Average",
            "detail": f"You study {sh} hrs/week vs class average: {ca_sh} hrs.",
            "action": ("Increase study time by 30 mins per day gradually. "
                       "Track your hours with a simple study log or journal.")
        })
    elif sh >= 20:
        strengths.append(f"High study dedication: {sh} hrs/week")

    #  7. CGPA 
    cgpa    = float(s.get("CGPA", 4))
    ca_cgpa = CLASS_AVG["CGPA"]
    if cgpa < 2.0:
        alerts.append(
            f"CGPA is critically low ({cgpa}). "
            "Academic probation may apply below 2.0 — speak to your academic advisor urgently."
        )
        recs.append({
            "icon":   "📊",
            "area":   "CGPA — Critical (below 2.0)",
            "detail": (f"Your CGPA: {cgpa} vs class average: {ca_cgpa}. "
                       "Sustained low CGPA risks academic dismissal."),
            "action": ("Consult your academic advisor about a CGPA recovery plan. "
                       "Focus maximum effort on currently enrolled courses to secure passing grades.")
        })
    elif cgpa < ca_cgpa - 0.5:
        recs.append({
            "icon":   "📊",
            "area":   "CGPA — Below Class Average",
            "detail": f"Your CGPA: {cgpa} vs class average: {ca_cgpa}.",
            "action": ("Prioritize current semester performance — each A grade raises CGPA significantly. "
                       "Avoid course overloads; quality over quantity this semester.")
        })
    elif cgpa >= 3.5:
        strengths.append(f"Strong CGPA: {cgpa}")

    #  8. Lab Work
    lb    = float(s.get("Average_Lab", 100))
    ca_lb = CLASS_AVG["Average_Lab"]
    if lb < 40:
        recs.append({
            "icon":   "🔬",
            "area":   "Lab Work — Needs Attention (below 40)",
            "detail": f"Lab avg: {lb} vs class average: {ca_lb}.",
            "action": ("Read the lab manual before each practical session. "
                       "Submit all pending lab reports — even late submissions may earn partial marks.")
        })
    elif lb < ca_lb - 10:
        recs.append({
            "icon":   "🔬",
            "area":   "Lab Work — Below Class Average",
            "detail": f"Lab avg: {lb} vs class average: {ca_lb}.",
            "action": ("Ask your lab instructor for feedback on weak areas. "
                       "Practice lab tasks outside class hours to build confidence.")
        })
    elif lb >= 80:
        strengths.append(f"Strong lab work: {lb}/100")

    #  9. Project Score 
    pj    = float(s.get("Project_Score", 100))
    ca_pj = CLASS_AVG["Project_Score"]
    if pj < 40:
        recs.append({
            "icon":   "💼",
            "area":   "Project Score — Below Standard (below 40)",
            "detail": f"Project: {pj} vs class average: {ca_pj}.",
            "action": ("Review the marking rubric — most marks are lost on documentation, not implementation. "
                       "Ensure all deliverables (reports, diagrams, testing) are complete.")
        })
    elif pj < ca_pj - 10:
        recs.append({
            "icon":   "💼",
            "area":   "Project Score — Below Class Average",
            "detail": f"Project: {pj} vs class average: {ca_pj}.",
            "action": ("Improve project documentation and presentation quality. "
                       "Discuss marking criteria with your supervisor before final submission.")
        })
    elif pj >= 80:
        strengths.append(f"Strong project: {pj}/100")

    # 10.  Extra Classes Alert (Phase 4 requirement) 
    if fail_prob >= 30:
        recs.append({
            "icon":   "🏫",
            "area":   "Extra Classes & Academic Support",
            "detail": (f"Your predicted At-Risk probability is {fail_prob:.1f}%. "
                       "Attending extra/remedial sessions significantly improves outcomes."),
            "action": ("Contact your course coordinator to enroll in extra tutorial sessions. "
                       "Visit the academic support center for one-on-one help. "
                       "Form or join a weekly study group of 3–4 motivated classmates.")
        })

    #  Positive fallback for strong students 
    if not recs and fail_prob < 30:
        recs.append({
            "icon":   "✅",
            "area":   "Outstanding — Performing Well Across All 10 Dimensions!",
            "detail": ("Your performance is above average in attendance, assignments, quizzes, "
                       "midterm, final exam, study hours, CGPA, lab work, and projects."),
            "action": ("Maintain your habits. Consider mentoring weaker classmates — "
                       "teaching reinforces your own understanding and builds leadership skills.")
        })

    return {
        "recs":       recs,
        "alerts":     alerts,
        "strengths":  strengths,
        "risk_level": rl,
        "risk_color": rc,
        "risk_bg":    rb,
        "fail_prob":  fail_prob,
    }

#  Helper Functions 
def perf_pill(val):
    m = {"Excellent": "pill-exc", "Very Good": "pill-vg", "Good": "pill-good",
         "Average": "pill-avg", "At-Risk": "pill-risk"}
    return f'<span class="pill {m.get(str(val), "pill-avg")}">{val}</span>'

def risk_label(fp):
    if fp >= 70: return '<span class="risk-critical">🔴 Critical</span>'
    if fp >= 50: return '<span class="risk-high">🟠 High</span>'
    if fp >= 30: return '<span class="risk-medium">🔵 Medium</span>'
    return '<span style="color:#48bb78;font-weight:700">🟢 Low</span>'

def require_login(role):
    if "role" not in session or session["role"] != role:
        return redirect(url_for("home"))
    return None

#  Routes 

@app.route("/")
def home():
    return render_template("login.html", last_id=1000 + total_students - 1, error=None)

@app.route("/login", methods=["POST"])
def login():
    role = request.form.get("role")
    pwd  = request.form.get("password")
    sid  = request.form.get("student_id", "").strip()
    last = 1000 + total_students - 1

    if role == "teacher" and pwd == "teacher123":
        session.clear()
        session["role"] = "teacher"
        return redirect(url_for("teacher_dashboard"))

    if role == "student" and pwd == "student123":
        if not sid:
            return render_template("login.html", last_id=last,
                                   error="Please enter your Student ID.")
        if df_final[df_final["Student_ID"] == sid].empty:
            return render_template("login.html", last_id=last,
                                   error=f"'{sid}' not found. Valid IDs: S1000 to S{last}")
        try:
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT IGNORE INTO users (username, password, role, student_id) "
                    "VALUES (%s, %s, %s, %s)",
                    (sid, "student123", "student", sid)
                )
            conn.commit()
            conn.close()
        except Exception:
            pass  # DB unavailable — still allow login

        session.clear()
        session["role"] = "student"
        session["student_id"] = sid
        return redirect(url_for("student_dashboard", student_id=sid))

    return render_template("login.html", last_id=last,
                           error="Wrong credentials. Please try again.")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/download/at-risk")
def download_at_risk():
    auth = require_login("teacher")
    if auth: return auth
    buf = io.StringIO()
    df_final[df_final["Final_Result"] == "Fail"].to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name="at_risk_students.csv")

@app.route("/download/all")
def download_all():
    auth = require_login("teacher")
    if auth: return auth
    buf = io.StringIO()
    df_final.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name="all_students_predictions.csv")

#  Teacher Dashboard 
@app.route("/dashboard/teacher")
def teacher_dashboard():
    auth = require_login("teacher")
    if auth: return auth

    pass_rate  = round(pass_count / total_students * 100, 1)
    fail_rate  = round(fail_count / total_students * 100, 1)
    avg_cgpa   = round(df_final["CGPA"].mean(), 2)
    avg_attend = round(df_final["Attendance_Percentage"].mean(), 1)
    avg_final  = round(df_final["Final_Exam"].mean(), 1)

    cfg  = {"displayModeBar": False}
    base = dict(paper_bgcolor="white", plot_bgcolor="white",
                showlegend=False, margin=dict(t=20, b=10, l=10, r=10), height=265)

    # Pass/Fail Pie
    f1 = go.Figure(go.Pie(
        labels=["Pass", "Fail"], values=[pass_count, fail_count],
        marker_colors=["#48bb78", "#fc8181"],
        hole=0.52, textinfo="percent+label", textfont_size=13))
    f1.update_layout(**{**base, "margin": dict(t=10, b=10, l=10, r=10)})

    # Performance Distribution Bar
    pc = df_final["Performance"].value_counts()
    cm = {"Excellent": "#48bb78", "Very Good": "#63b3ed", "Good": "#f6ad55",
          "Average": "#e9d8fd", "At-Risk": "#fc8181"}
    f2 = go.Figure(go.Bar(
        x=list(pc.index), y=list(pc.values),
        marker_color=[cm.get(p, "#999") for p in pc.index],
        text=list(pc.values), textposition="outside"))
    f2.update_layout(**{**base, "yaxis": dict(gridcolor="#f0f4ff", zeroline=False, title="Students")})

    # Score breakdown bar
    sc = ["Average_Assignment", "Average_Quiz", "Midterm_Exam", "Final_Exam", "Average_Lab", "Project_Score"]
    sl = ["Assignment", "Quiz", "Midterm", "Final Exam", "Lab Work", "Project"]
    f3 = go.Figure(go.Bar(
        x=sl, y=[round(df_final[c].mean(), 1) for c in sc],
        marker_color=["#667eea", "#764ba2", "#f6ad55", "#fc8181", "#48bb78", "#63b3ed"],
        text=[str(round(df_final[c].mean(), 1)) for c in sc], textposition="outside"))
    f3.update_layout(**{**base, "yaxis": dict(gridcolor="#f0f4ff", zeroline=False, title="Avg Score")})

    # df_final stores encoded numbers — decode back using feat_encoders
    df_display = df_final.copy()
    if "Gender" in df_display.columns and "Gender" in feat_encoders:
        le_g = feat_encoders["Gender"]
        def safe_decode_gender(v):
            try:
                vi = int(float(str(v)))
                return le_g.classes_[vi] if 0 <= vi < len(le_g.classes_) else str(v)
            except Exception:
                return str(v) if str(v) in le_g.classes_ else str(v)
        df_display["Gender"] = df_display["Gender"].apply(safe_decode_gender)

    if "Course" in df_display.columns and "Course" in feat_encoders:
        le_c = feat_encoders["Course"]
        def safe_decode_course(v):
            try:
                vi = int(float(str(v)))
                return le_c.classes_[vi] if 0 <= vi < len(le_c.classes_) else str(v)
            except Exception:
                return str(v) if str(v) in le_c.classes_ else str(v)
        df_display["Course"] = df_display["Course"].apply(safe_decode_course)

    # Score breakdown bar (3rd chart — kept)
    sc = ["Average_Assignment", "Average_Quiz", "Midterm_Exam", "Final_Exam", "Average_Lab", "Project_Score"]
    sl = ["Assignment", "Quiz", "Midterm", "Final Exam", "Lab Work", "Project"]
    f3 = go.Figure(go.Bar(
        x=sl, y=[round(df_final[c].mean(), 1) for c in sc],
        marker_color=["#667eea", "#764ba2", "#f6ad55", "#fc8181", "#48bb78", "#63b3ed"],
        text=[str(round(df_final[c].mean(), 1)) for c in sc], textposition="outside"))
    f3.update_layout(**{**base, "yaxis": dict(gridcolor="#f0f4ff", zeroline=False, title="Avg Score")})

    # Table columns — Registration_No added, Section removed, Gender/Course now decoded
    dc   = ["Student_ID", "Registration_No", "Gender", "Age", "Course", "Semester",
            "Attendance_Percentage", "Average_Assignment", "Average_Quiz",
            "Midterm_Exam", "Final_Exam", "CGPA", "Study_Hours_Per_Week",
            "Performance", "Final_Result", "Fail_Probability"]
    # Registration_No may not be in df_display if it was dropped — add it from df_full
    if "Registration_No" not in df_display.columns and "Registration_No" in df_full.columns:
        df_display = df_display.merge(
            df_full[["Student_ID", "Registration_No"]], on="Student_ID", how="left")
    cols = [c for c in dc if c in df_display.columns]
    hdr  = "".join(f"<th>{c.replace('_',' ')}</th>" for c in cols) + "<th>Risk Level</th>"

    def build_rows(df_rows, is_risk=False):
        rows = ""
        for _, row in df_rows[cols].iterrows():
            rows += "<tr>"
            for c in cols:
                v = row[c]
                if c == "Final_Result":
                    cls = "pill-fail" if v == "Fail" else "pill-pass"
                    rows += f'<td><span class="pill {cls}">{v}</span></td>'
                elif c == "Performance":
                    rows += f"<td>{perf_pill(v)}</td>"
                elif c == "Final_Exam" and is_risk and pd.to_numeric(v, errors="coerce") < 50:
                    rows += f'<td style="color:#fc8181;font-weight:800">{v}</td>'
                else:
                    rows += f"<td>{v}</td>"
            rows += f"<td>{risk_label(float(row.get('Fail_Probability', 0)))}</td></tr>"
        return rows

    df_risk = df_display[df_display["Final_Result"] == "Fail"].sort_values(
        "Fail_Probability", ascending=False)

    return render_template("teacher.html",
        model_name   = type(model).__name__,
        total        = total_students,
        pass_count   = pass_count,
        fail_count   = fail_count,
        pass_rate    = pass_rate,
        fail_rate    = fail_rate,
        avg_final    = avg_final,
        avg_attend   = avg_attend,
        avg_cgpa     = avg_cgpa,
        chart_pie    = f1.to_html(full_html=False, config=cfg),
        chart_perf   = f2.to_html(full_html=False, config=cfg),
        chart_scores = f3.to_html(full_html=False, config=cfg),
        header_html  = hdr,
        risk_rows    = build_rows(df_risk, is_risk=True),
        all_rows     = build_rows(df_display.sort_values("Student_ID")))

#  Student Dashboard 
@app.route("/dashboard/student/<student_id>")
def student_dashboard(student_id):
    auth = require_login("student")
    if auth: return auth
    if session.get("student_id") != student_id:
        return redirect(url_for("student_dashboard",
                                student_id=session.get("student_id", "")))
    try:
        s = df_final[df_final["Student_ID"] == student_id].iloc[0]
    except IndexError:
        return render_template("login.html", last_id=1000 + total_students - 1,
                               error=f"Student '{student_id}' not found.")

    predicted   = s["Final_Result"]
    performance = s.get("Performance", "N/A")
    is_pass     = (predicted == "Pass")
    fail_prob   = float(s.get("Fail_Probability", 0))

    outcome_class = "outcome-pass" if is_pass else "outcome-fail"
    outcome_text  = "✓ Predicted to PASS — Keep it up!" if is_pass else "⚠ Predicted to FAIL — Immediate Action Required"
    avatar_bg     = "#c6f6d5" if is_pass else "#fed7d7"
    avatar_color  = "#276749" if is_pass else "#9b2335"
    avatar_emoji  = "😊" if is_pass else "📚"

    # Score comparison chart
    sc = ["Average_Assignment", "Average_Quiz", "Midterm_Exam", "Final_Exam", "Average_Lab", "Project_Score"]
    sl = ["Assignment", "Quiz", "Midterm", "Final Exam", "Lab Work", "Project"]
    ss = [round(float(s[c]), 1) for c in sc if c in df_final.columns]
    ca = [round(df_final[c].mean(), 1) for c in sc if c in df_final.columns]

    fig = go.Figure([
        go.Bar(name="Your Score",    x=sl, y=ss, marker_color="#667eea",
               text=[str(v) for v in ss], textposition="outside"),
        go.Bar(name="Class Average", x=sl, y=ca, marker_color="#f6ad55",
               opacity=0.75, text=[str(v) for v in ca], textposition="outside")
    ])
    fig.update_layout(barmode="group", paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(gridcolor="#f0f4ff", zeroline=False, title="Score"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=30, b=10, l=10, r=10), height=300)
    chart = fig.to_html(full_html=False, config={"displayModeBar": False})

    # Score progress bars
    bars_html = ""
    for lbl, score, avg in zip(sl, ss, ca):
        d   = score - avg
        sgn = "+" if d >= 0 else ""
        cls = "diff-up" if d >= 0 else "diff-down"
        bars_html += f"""<div class="score-row">
          <div class="score-label">
            <span>{lbl}</span>
            <span>{score}&nbsp;<span class="{cls} score-diff">{sgn}{d:.1f} vs class</span></span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:{min(score,100)}%"></div></div>
        </div>"""

    # Gender label
    gender_raw = str(s.get("Gender", ""))
    gender     = gender_raw if gender_raw in ("Male", "Female") else ("Female" if gender_raw == "0" else "Male")

    # Recommendations
    rd = generate_recommendations(s)

    alerts_html = ""
    for a in rd["alerts"]:
        cls = "alert-critical" if any(w in a.lower() for w in ["critical", "probation", "barred"]) else "alert-high"
        alerts_html += (f'<div class="alert-banner {cls}">'
                        f'<div class="alert-title">⚠ Early Alert</div>'
                        f'<div class="alert-text">{a}</div></div>')

    rec_cards_html = ""
    for r in rd["recs"]:
        rec_cards_html += f"""<div class="rec-card">
          <div class="rec-card-icon">{r['icon']}</div>
          <div class="rec-card-area">{r['area']}</div>
          <div class="rec-card-detail">{r['detail']}</div>
          <div class="rec-card-action">{r['action']}</div>
        </div>"""

    strengths_html = ""
    if rd["strengths"]:
        pills = "".join(f'<span class="strength-pill">{x}</span>' for x in rd["strengths"])
        strengths_html = f"""<div class="chart-card" style="margin-bottom:18px">
          <div class="chart-title">💪 Your Strengths</div>
          <div class="strength-row">{pills}</div>
        </div>"""

    # Decode course name from encoded number
    course_raw = s.get("Course", "N/A")
    if "Course" in feat_encoders:
        try:
            ci = int(float(str(course_raw)))
            course_display = feat_encoders["Course"].classes_[ci]
        except Exception:
            course_display = str(course_raw)
    else:
        course_display = str(course_raw)

    # Registration No
    reg_no = s.get("Registration_No", "N/A") if "Registration_No" in s.index else "N/A"
    # Try from df_full if not in df_final
    if str(reg_no) in ("N/A", "nan", ""):
        try:
            reg_no = df_full[df_full["Student_ID"] == student_id]["Registration_No"].iloc[0]
        except Exception:
            reg_no = "N/A"

    return render_template("student.html",
        student_id     = student_id,
        reg_no         = reg_no,
        gender         = gender,
        age            = s.get("Age", "N/A"),
        course         = course_display,
        semester       = s.get("Semester", "N/A"),
        cgpa           = s.get("CGPA", "N/A"),
        prev_gpa       = s.get("Previous_GPA", "N/A"),
        study_hrs      = s.get("Study_Hours_Per_Week", "N/A"),
        attend         = s.get("Attendance_Percentage", "N/A"),
        lib            = s.get("Library_Visits", "N/A"),
        online         = s.get("Online_Course_Hours", "N/A"),
        total_score    = round(float(s.get("Total_Score", 0)), 1),
        performance    = performance,
        predicted      = predicted,
        fail_prob      = fail_prob,
        outcome_class  = outcome_class,
        outcome_text   = outcome_text,
        avatar_bg      = avatar_bg,
        avatar_color   = avatar_color,
        avatar_emoji   = avatar_emoji,
        chart          = chart,
        bars_html      = bars_html,
        risk_level     = rd["risk_level"],
        risk_color     = rd["risk_color"],
        risk_bg        = rd["risk_bg"],
        alerts_html    = alerts_html,
        rec_cards_html = rec_cards_html,
        strengths_html = strengths_html)

#  Start Server 
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  CS619 — Student Performance Predictor  |  Phase 4 Final")
    print("  Authors: Syeda Farkhanda Batool Naqvi & Saad Shakir")
    print("=" * 65)
    print("  Open browser ──▶  http://127.0.0.1:5000")
    print("  Teacher login:    password = teacher123")
    print(f"  Student login:    password = student123  +  Student ID")
    print(f"  Valid IDs:        S1000 to S{1000 + total_students - 1}")
    print("=" * 65 + "\n")
    app.run(debug=False, port=5000)