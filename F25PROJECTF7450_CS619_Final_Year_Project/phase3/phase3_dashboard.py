# CS619 - Phase 3: Student Performance Dashboard
# Dataset: Student Academic Performance (2000 students)
# Tools Used: Flask, Pandas, Plotly, Joblib
# Flask is a lightweight Python web framework for building web apps
# and handles different pages (routes) of the dashboard

from flask import Flask, request, redirect, url_for, send_file

import pandas as pd

import plotly.graph_objects as go

import joblib

import numpy as np

import io

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


# STEP 1: Load Dataset and All Phase 2 Files

df_full      = pd.read_csv("processed_student_data.csv")
model        = joblib.load("student_model_v1.pkl")
scaler       = joblib.load("student_scaler_v1.pkl")
target_encoder = joblib.load("student_encoder_v1.pkl")
le_dict      = joblib.load("student_feature_encoders_v1.pkl")

_model_name  = type(model).__name__
_n_students  = len(df_full)


# FIX 1 & 2: Clean dirty data rows before prediction
# 5 students had Attendance_Percentage = 200 (calculation error in raw data)
# This cascaded into Total_Score = 150 (impossible value)
# We recalculate both columns for affected rows.

bad_attend = df_full["Attendance_Percentage"] > 100
if bad_attend.any():
    df_full.loc[bad_attend, "Attendance_Percentage"] = (
        df_full.loc[bad_attend, "Classes_Attended"] /
        df_full.loc[bad_attend, "Total_Classes"] * 100
    ).round(2)

bad_total = df_full["Total_Score"] > 100
if bad_total.any():
    df_full.loc[bad_total, "Total_Score"] = (
        (df_full.loc[bad_total, "Average_Assignment"] +
         df_full.loc[bad_total, "Average_Quiz"] +
         df_full.loc[bad_total, "Midterm_Exam"] +
         df_full.loc[bad_total, "Final_Exam"] +
         df_full.loc[bad_total, "Average_Lab"] +
         df_full.loc[bad_total, "Project_Score"] +
         df_full.loc[bad_total, "Attendance_Percentage"]) / 7
    ).round(2)


# STEP 2: Define Feature Lists
# These MUST match exactly what Phase 2 used

DROP_COLS = ["Performance", "Student_ID", "Registration_No"]

# Categorical columns that Phase 2 encoded (NOT Student_ID/Registration_No — those are dropped)
CATEGORICAL_COLS = ["Gender", "Course", "Section", "Sports_Activity", "Club_Membership"]

NUMERIC_COLS = [
    'Age','Semester','Total_Classes','Classes_Attended','Attendance_Percentage',
    'Assignment_1','Assignment_2','Assignment_3','Average_Assignment',
    'Quiz_1','Quiz_2','Average_Quiz','Midterm_Exam','Final_Exam',
    'Lab_Work_1','Lab_Work_2','Average_Lab','Project_Score','CGPA','Previous_GPA',
    'Study_Hours_Per_Week','Library_Visits','Online_Course_Hours',
    'Class_Participation','Group_Work_Score','Total_Score',
]

SCORE_COLS   = ["Average_Assignment","Average_Quiz","Midterm_Exam",
                "Final_Exam","Average_Lab","Project_Score"]
SCORE_LABELS = ["Assignment","Quiz","Midterm","Final Exam","Lab Work","Project"]


# STEP 3: Generate Predictions for All Students

def predict_all_students(data):
    """
    Replicates the exact Phase 2 preprocessing pipeline:
    1. Label-encode categoricals using saved le_dict
    2. Drop non-feature columns
    3. Scale with the saved StandardScaler
    4. Predict with the saved model
    5. Decode predictions back to category names
    """
    df_copy = data.copy()

    # Step A – encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df_copy.columns and col in le_dict:
            le = le_dict[col]
            df_copy[col] = (
                df_copy[col].astype(str)
                .map(lambda x: x if x in le.classes_ else le.classes_[0])
            )
            df_copy[col] = le.transform(df_copy[col])

    # Step B – build feature matrix
    existing_drop = [c for c in DROP_COLS if c in df_copy.columns]
    X = df_copy.drop(columns=existing_drop)

    # Step C – coerce to numeric, fill NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())

    # Step D – scale only the numeric columns (same ones Phase 2 used)
    num_present = [c for c in NUMERIC_COLS if c in X.columns]
    X[num_present] = scaler.transform(X[num_present])

    # Step E – predict & decode
    raw_preds = model.predict(X.values)
    readable  = target_encoder.inverse_transform(raw_preds)
    data["Predicted_Performance"] = readable

    # Step F – Pass / Fail
    # FIX 3: "Average" category students should NOT be marked as Fail.
    # Original code only included Excellent/Very Good/Good → Average was wrongly
    # treated as Fail. "At-Risk" remains the only Fail/at-risk category.
    PASS_CATS = {"Excellent", "Very Good", "Good", "Average"}
    data["Predicted_Result"] = data["Predicted_Performance"].apply(
        lambda p: "Pass" if p in PASS_CATS else "Fail"
    )

    # Step G – round floats
    float_cols = data.select_dtypes(include=["float64","float32"]).columns
    data[float_cols] = data[float_cols].round(2)
    return data


df_final = predict_all_students(df_full.copy())

total_students = len(df_final)
pass_count = int((df_final["Predicted_Result"] == "Pass").sum())
fail_count = int((df_final["Predicted_Result"] == "Fail").sum())

print()
print("=" * 58)
print("   CS619 — Student Performance Prediction System")
print("   Phase 3 Dashboard  |  ML Model: Logistic Regression")
print("=" * 58)
print(f"   Students loaded   : {_n_students:,}")
print(f"   Predicted Pass    : {pass_count:,}  ({round(pass_count/_n_students*100,1)}%)")
print(f"   Predicted At-Risk : {fail_count:,}  ({round(fail_count/_n_students*100,1)}%)")
print("=" * 58)
print("   Dashboard running at  http://127.0.0.1:5000")
print("   Teacher login  →  password: teacher123")
print("   Student login  →  password: student123")
print("=" * 58)
print()


# SHARED CSS STYLES — Purple gradient theme
# (matches sample dashboard exactly)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Nunito', sans-serif;
    background: #f0f4ff;
    min-height: 100vh;
}

/* ---- NAVBAR ---- */
.navbar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px 30px;
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    position: sticky; top: 0; z-index: 100;
}
.navbar-title { color: white; font-size: 1.25rem; font-weight: 800; }
.navbar-title span {
    background: rgba(255,255,255,0.2); padding: 3px 10px;
    border-radius: 20px; font-size: 0.78rem; margin-left: 10px; font-weight: 600;
}
.logout-link {
    color: rgba(255,255,255,0.9); text-decoration: none; font-size: 0.88rem;
    font-weight: 700; background: rgba(255,255,255,0.15);
    padding: 7px 18px; border-radius: 20px; transition: background 0.2s;
}
.logout-link:hover { background: rgba(255,255,255,0.28); }

/* ---- PAGE WRAPPER ---- */
.page { max-width: 1350px; margin: 0 auto; padding: 28px 20px; }
.section-heading { font-size: 1.55rem; font-weight: 900; color: #2d3748; margin-bottom: 4px; }
.section-sub { color: #718096; font-size: 0.88rem; margin-bottom: 24px; }

/* ---- STAT CARDS ---- */
.stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 14px; margin-bottom: 26px;
}
.stat-card {
    background: white; border-radius: 16px; padding: 18px 16px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07); border-left: 5px solid;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-3px); }
.stat-card.purple { border-color: #667eea; }
.stat-card.green  { border-color: #48bb78; }
.stat-card.red    { border-color: #fc8181; }
.stat-card.orange { border-color: #f6ad55; }
.stat-card.blue   { border-color: #63b3ed; }
.stat-card.pink   { border-color: #f687b3; }
.stat-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; color: #a0aec0; font-weight: 700; margin-bottom: 7px; }
.stat-value { font-size: 1.9rem; font-weight: 900; line-height: 1; }
.stat-card.purple .stat-value { color: #667eea; }
.stat-card.green  .stat-value { color: #48bb78; }
.stat-card.red    .stat-value { color: #fc8181; }
.stat-card.orange .stat-value { color: #f6ad55; }
.stat-card.blue   .stat-value { color: #63b3ed; }
.stat-card.pink   .stat-value { color: #f687b3; }
.stat-note { font-size: 0.76rem; color: #cbd5e0; margin-top: 3px; }

/* ---- CHART CARDS ---- */
.charts-row {
    display: grid; grid-template-columns: 1fr 1.6fr;
    gap: 18px; margin-bottom: 20px;
}
.chart-card {
    background: white; border-radius: 16px;
    padding: 20px; box-shadow: 0 3px 12px rgba(0,0,0,0.07);
}
.chart-title { font-size: 0.92rem; font-weight: 800; color: #4a5568; margin-bottom: 14px; }

/* ---- TABS ---- */
.tabs { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
.tab-btn {
    padding: 8px 20px; border-radius: 25px;
    border: 2px solid #e2e8f0; background: white;
    color: #718096; font-family: 'Nunito', sans-serif;
    font-size: 0.82rem; font-weight: 700; cursor: pointer; transition: all 0.2s;
}
.tab-btn.active {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-color: transparent; color: white;
}

/* ---- TABLE ---- */
.table-card {
    background: white; border-radius: 16px; padding: 20px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07); overflow-x: auto;
}
.table-header-row {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 10px;
}
.table-title { font-size: 0.98rem; font-weight: 800; color: #2d3748; }
table { width: 100%; border-collapse: collapse; font-size: 0.81rem; }
th {
    background: #f7fafc; padding: 10px 11px; text-align: left;
    font-weight: 800; color: #4a5568; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 2px solid #e2e8f0; white-space: nowrap;
}
td { padding: 9px 11px; color: #4a5568; border-bottom: 1px solid #f0f4ff; white-space: nowrap; }
tr:hover td { background: #f7faff; }

/* ---- PILLS ---- */
.pill { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 800; }
.pill-pass { background: #c6f6d5; color: #276749; }
.pill-fail { background: #fed7d7; color: #9b2335; }
.pill-exc  { background: #bee3f8; color: #2a69ac; }
.pill-vg   { background: #c6f6d5; color: #276749; }
.pill-good { background: #fefcbf; color: #7b6f00; }
.pill-avg  { background: #c6f6d5; color: #276749; }
.pill-risk { background: #fed7d7; color: #9b2335; }

/* ---- DOWNLOAD BUTTON ---- */
.dl-btn {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; text-decoration: none; padding: 7px 16px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700;
    transition: opacity 0.2s; white-space: nowrap;
}
.dl-btn:hover { opacity: 0.85; }

/* ---- STUDENT PAGE ---- */
.student-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 18px; padding: 26px 28px; margin-bottom: 22px;
    color: white; display: flex; align-items: center; gap: 22px; flex-wrap: wrap;
}
.avatar-circle {
    width: 72px; height: 72px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.9rem; flex-shrink: 0;
    border: 3px solid rgba(255,255,255,0.4);
}
.banner-name { font-size: 1.4rem; font-weight: 900; }
.banner-sub  { font-size: 0.85rem; opacity: 0.82; margin-top: 4px; line-height: 1.7; }
.outcome-tag {
    margin-top: 10px; display: inline-block;
    padding: 5px 16px; border-radius: 25px; font-weight: 800; font-size: 0.9rem;
}
.outcome-pass { background: #c6f6d5; color: #276749; }
.outcome-fail { background: #fed7d7; color: #9b2335; }

.two-col { display: grid; grid-template-columns: 1fr 1.8fr; gap: 18px; margin-bottom: 18px; }

.mini-stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px; margin-bottom: 18px;
}
.mini-card {
    background: white; border-radius: 14px; padding: 14px 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.06); text-align: center;
}
.mini-val { font-size: 1.4rem; font-weight: 900; color: #667eea; }
.mini-lbl { font-size: 0.7rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 3px; }

.score-row { margin-bottom: 13px; }
.score-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; font-weight: 700; color: #4a5568; margin-bottom: 5px;
}
.score-diff { font-size: 0.72rem; font-weight: 600; }
.diff-up   { color: #48bb78; }
.diff-down { color: #fc8181; }
.bar-track { background: #edf2f7; border-radius: 10px; height: 9px; overflow: hidden; }
.bar-fill  { height: 9px; border-radius: 10px; background: linear-gradient(90deg, #667eea, #764ba2); display: block; }

/* ---- SEARCH BOX ---- */
.search-box {
    padding: 8px 14px; border: 2px solid #e2e8f0; border-radius: 20px;
    font-family: 'Nunito', sans-serif; font-size: 0.85rem; outline: none;
    transition: border-color 0.2s; width: 220px;
}
.search-box:focus { border-color: #667eea; }

/* ---- ALERT BANNER ---- */
.alert-banner {
    background: linear-gradient(135deg, #fed7d7, #feb2b2);
    border: 1px solid #fc8181; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 12px;
    font-size: 0.88rem; color: #9b2335; font-weight: 700;
}

/* ---- LOGIN PAGE ---- */
.login-page {
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.login-box {
    background: white; border-radius: 22px; padding: 38px 34px;
    width: 100%; max-width: 410px;
    box-shadow: 0 25px 60px rgba(102,126,234,0.35);
}
.login-logo { text-align: center; margin-bottom: 26px; }
.app-name {
    font-size: 1.45rem; font-weight: 900;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-sub { font-size: 0.8rem; color: #a0aec0; margin-top: 3px; }
.field { margin-bottom: 16px; }
label {
    display: block; font-size: 0.78rem; font-weight: 700;
    color: #718096; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
}
select, input[type="text"], input[type="password"] {
    width: 100%; padding: 11px 13px; border: 2px solid #e2e8f0;
    border-radius: 11px; font-family: 'Nunito', sans-serif;
    font-size: 0.9rem; color: #2d3748; outline: none;
    transition: border-color 0.2s; background: #f7fafc;
}
select:focus, input:focus { border-color: #667eea; background: white; }
.login-btn {
    width: 100%; padding: 12px; border: none; border-radius: 11px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; font-family: 'Nunito', sans-serif;
    font-size: 0.98rem; font-weight: 800; cursor: pointer; margin-top: 6px; transition: opacity 0.2s;
}
.login-btn:hover { opacity: 0.9; }
.hint-box {
    margin-top: 18px; background: #f0f4ff; border-radius: 11px;
    padding: 13px 15px; font-size: 0.8rem; color: #718096; line-height: 1.8;
}
.hint-box b { color: #667eea; }
"""


# Helper: Performance pill HTML


def perf_pill(val):
    val = str(val)
    if val == "Excellent": return f'<span class="pill pill-exc">{val}</span>'
    if val == "Very Good":  return f'<span class="pill pill-vg">{val}</span>'
    if val == "Good":       return f'<span class="pill pill-good">{val}</span>'
    if val == "Average":    return f'<span class="pill pill-avg">{val}</span>'
    if val == "At-Risk":    return f'<span class="pill pill-risk">{val}</span>'
    return val


# ROUTE 1: Login Page


@app.route("/")
def home():
    last_id = 1000 + total_students - 1
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>CS619 — StudentPredictor Login</title>
  <style>{CSS}</style>
</head>
<body>
<div class="login-page">
  <div class="login-box">
    <div class="login-logo">
      <div class="app-name">📊 StudentPredictor</div>
      <div class="app-sub">CS619 · ML Dashboard · Phase 3</div>
    </div>

    <form method="POST" action="/login">
      <div class="field">
        <label>Select Role</label>
        <select name="role" id="role" onchange="toggleID(this.value)">
          <option value="teacher">👩‍🏫 Teacher</option>
          <option value="student">🎓 Student</option>
        </select>
      </div>

      <div class="field">
        <label>Password</label>
        <input type="password" name="password" placeholder="Enter your password" required>
      </div>

      <div class="field" id="id-field" style="display:none;">
        <label>Student ID (S1000 to S{last_id})</label>
        <input type="text" name="student_id" placeholder="e.g. S1042">
      </div>

      <button type="submit" class="login-btn">Sign In →</button>
    </form>

    <div class="hint-box">
      <b>Teacher:</b> password = teacher123<br>
      <b>Student:</b> password = student123<br>
      <b>Student ID:</b> S1000 to S{last_id}
    </div>
  </div>
</div>
<script>
function toggleID(val) {{
  document.getElementById('id-field').style.display = val === 'student' ? 'block' : 'none';
}}
</script>
</body></html>"""


# ROUTE 2: Handle Login


@app.route("/login", methods=["POST"])
def login():
    role     = request.form.get("role")
    password = request.form.get("password")
    sid      = request.form.get("student_id", "").strip()

    if role == "teacher" and password == "teacher123":
        return redirect(url_for("teacher_dashboard"))
    elif role == "student" and password == "student123":
        if not sid:
            return "Please enter your Student ID.", 400
        return redirect(url_for("student_dashboard", student_id=sid))
    else:
        return "Invalid credentials. Please go back and try again.", 401


# ROUTE 3: Download At-Risk CSV


@app.route("/download/at-risk")
def download_at_risk():
    df_risk = df_final[df_final["Predicted_Result"] == "Fail"].copy()
    buf = io.StringIO()
    df_risk.to_csv(buf, index=False); buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype="text/csv", as_attachment=True,
                     download_name="at_risk_students.csv")


# ROUTE 4: Download All Students CSV


@app.route("/download/all")
def download_all():
    buf = io.StringIO()
    df_final.to_csv(buf, index=False); buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype="text/csv", as_attachment=True,
                     download_name="all_students_predictions.csv")


# ROUTE 5: Teacher Dashboard


@app.route("/dashboard/teacher")
def teacher_dashboard():
    pass_rate  = round(pass_count / total_students * 100, 1)
    avg_attend = round(df_final["Attendance_Percentage"].mean(), 1)
    avg_final  = round(df_final["Final_Exam"].mean(), 1)
    avg_cgpa   = round(df_final["Previous_GPA"].mean(), 2)
    avg_study  = round(df_final["Study_Hours_Per_Week"].mean(), 1)
    risk_rate  = round(fail_count / total_students * 100, 1)

    # --- CHART 1: Pass vs Fail donut ---
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Pass","Fail"],
        values=[pass_count, fail_count],
        marker_colors=["#48bb78","#fc8181"],
        hole=0.52, textinfo="percent+label", textfont_size=13
    )])
    fig_pie.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                          margin=dict(t=10,b=10,l=10,r=10), height=265, showlegend=False)
    chart_pie = fig_pie.to_html(full_html=False, config={"displayModeBar": False})

    # --- CHART 2: Performance distribution ---
    perf_order = ["Excellent","Very Good","Good","Average","At-Risk"]
    perf_counts = df_final["Performance"].value_counts().reindex(perf_order, fill_value=0)
    perf_color_map = {"Excellent":"#48bb78","Very Good":"#63b3ed",
                      "Good":"#f6ad55","Average":"#e9d8fd","At-Risk":"#fc8181"}
    fig_perf = go.Figure(data=[go.Bar(
        x=list(perf_counts.index), y=list(perf_counts.values),
        marker_color=[perf_color_map.get(p,"#999") for p in perf_counts.index],
        text=list(perf_counts.values), textposition="outside", textfont=dict(size=12)
    )])
    fig_perf.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                           yaxis=dict(gridcolor="#f0f4ff", zeroline=False, title="Number of Students"),
                           xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                           margin=dict(t=20,b=10,l=10,r=10), height=265, showlegend=False)
    chart_perf = fig_perf.to_html(full_html=False, config={"displayModeBar": False})

    # --- CHART 3: Average scores by subject ---
    score_vals = [round(df_final[c].mean(), 1) for c in SCORE_COLS]
    fig_scores = go.Figure(data=[go.Bar(
        x=SCORE_LABELS, y=score_vals,
        marker_color=["#667eea","#764ba2","#f6ad55","#fc8181","#48bb78","#63b3ed"],
        text=[f"{v}" for v in score_vals], textposition="outside", textfont=dict(size=11)
    )])
    fig_scores.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                             yaxis=dict(gridcolor="#f0f4ff", zeroline=False, title="Average Score"),
                             xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                             margin=dict(t=20,b=10,l=10,r=10), height=265, showlegend=False)
    chart_scores = fig_scores.to_html(full_html=False, config={"displayModeBar": False})

    # --- CHART 4: Gender distribution donut ---
    gender_counts = df_final["Gender"].value_counts()
    fig_gender = go.Figure(data=[go.Pie(
        labels=list(gender_counts.index), values=list(gender_counts.values),
        marker_colors=["#667eea","#f687b3","#63b3ed"],
        hole=0.52, textinfo="percent+label", textfont_size=12
    )])
    fig_gender.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                             margin=dict(t=10,b=10,l=10,r=10), height=230, showlegend=False)
    chart_gender = fig_gender.to_html(full_html=False, config={"displayModeBar": False})

    # --- CHART 5: Course-wise at-risk bar ---
    course_risk = df_final[df_final["Predicted_Result"]=="Fail"].groupby("Course").size().sort_values(ascending=False)
    fig_course = go.Figure(data=[go.Bar(
        x=list(course_risk.index), y=list(course_risk.values),
        marker_color="#fc8181",
        text=list(course_risk.values), textposition="outside", textfont=dict(size=11)
    )])
    fig_course.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                             yaxis=dict(gridcolor="#f0f4ff", zeroline=False, title="At-Risk Count"),
                             xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                             margin=dict(t=20,b=10,l=10,r=10), height=230, showlegend=False)
    chart_course = fig_course.to_html(full_html=False, config={"displayModeBar": False})

    # --- Table columns ---
    display_cols = [
        "Student_ID","Registration_No","Gender","Age","Course",
        "Semester","Section","Attendance_Percentage",
        "Average_Assignment","Average_Quiz","Midterm_Exam","Final_Exam",
        "Average_Lab","Project_Score","Previous_GPA",
        "Study_Hours_Per_Week","Sports_Activity","Club_Membership",
        "Total_Score","Performance","Predicted_Performance","Predicted_Result"
    ]
    cols = [c for c in display_cols if c in df_final.columns]
    header_html = "".join(f"<th>{c.replace('_',' ')}</th>" for c in cols)

    # At-risk rows
    df_risk = df_final[df_final["Predicted_Result"]=="Fail"].sort_values("Final_Exam")
    risk_rows = ""
    for _, row in df_risk[cols].iterrows():
        risk_rows += "<tr>"
        for c in cols:
            val = row[c]
            if c == "Predicted_Result":
                risk_rows += '<td><span class="pill pill-fail">✕ Fail</span></td>'
            elif c in ("Performance","Predicted_Performance"):
                risk_rows += f"<td>{perf_pill(val)}</td>"
            elif c == "Final_Exam":
                try:
                    risk_rows += (f'<td style="color:#fc8181;font-weight:800">{val}</td>'
                                  if float(val) < 50 else f"<td>{val}</td>")
                except: risk_rows += f"<td>{val}</td>"
            else:
                risk_rows += f"<td>{val}</td>"
        risk_rows += "</tr>"

    # All students rows
    all_rows = ""
    for _, row in df_final[cols].sort_values("Student_ID").iterrows():
        all_rows += "<tr>"
        for c in cols:
            val = row[c]
            if c == "Predicted_Result":
                if val == "Pass":
                    all_rows += '<td><span class="pill pill-pass">✓ Pass</span></td>'
                else:
                    all_rows += '<td><span class="pill pill-fail">✕ Fail</span></td>'
            elif c in ("Performance","Predicted_Performance"):
                all_rows += f"<td>{perf_pill(val)}</td>"
            else:
                all_rows += f"<td>{val}</td>"
        all_rows += "</tr>"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Teacher Dashboard — CS619</title>
  <style>{CSS}</style>
</head>
<body>

<div class="navbar">
  <div class="navbar-title">📊 StudentPredictor <span>Teacher View</span></div>
  <a href="/" class="logout-link">Logout</a>
</div>

<div class="page">

  <div class="section-heading">Class Performance Overview</div>
  <div class="section-sub">CS619 · {type(model).__name__} Model · {total_students} students enrolled · Pass = Excellent / Very Good / Good / Average &nbsp;|&nbsp; Fail = At-Risk only</div>

  <!-- Alert Banner -->
  {'<div class="alert-banner">⚠️ ' + str(fail_count) + ' students (' + str(risk_rate) + '%) are predicted to fail and need immediate attention!</div>' if fail_count > 0 else ''}

  <!-- Stat Cards -->
  <div class="stats-row">
    <div class="stat-card purple">
      <div class="stat-label">Total Students</div>
      <div class="stat-value">{total_students}</div>
      <div class="stat-note">Enrolled</div>
    </div>
    <div class="stat-card green">
      <div class="stat-label">Predicted Pass</div>
      <div class="stat-value">{pass_count}</div>
      <div class="stat-note">{pass_rate}% pass rate</div>
    </div>
    <div class="stat-card red">
      <div class="stat-label">At-Risk (Fail)</div>
      <div class="stat-value">{fail_count}</div>
      <div class="stat-note">Need attention ⚠️</div>
    </div>
    <div class="stat-card orange">
      <div class="stat-label">Avg Final Exam</div>
      <div class="stat-value">{avg_final}</div>
      <div class="stat-note">Class average</div>
    </div>
    <div class="stat-card blue">
      <div class="stat-label">Avg Attendance</div>
      <div class="stat-value">{avg_attend}%</div>
      <div class="stat-note">Class average</div>
    </div>
    <div class="stat-card pink">
      <div class="stat-label">Avg Study Hrs</div>
      <div class="stat-value">{avg_study}</div>
      <div class="stat-note">Per week</div>
    </div>
  </div>

  <!-- Charts Row 1: Pie + Performance -->
  <div class="charts-row">
    <div class="chart-card">
      <div class="chart-title">🎯 Pass / Fail Prediction Split</div>
      {chart_pie}
    </div>
    <div class="chart-card">
      <div class="chart-title">🏆 Performance Category Distribution</div>
      {chart_perf}
    </div>
  </div>

  <!-- Chart: Subject Averages -->
  <div class="chart-card" style="margin-bottom:20px;">
    <div class="chart-title">📈 Class Average Score by Subject</div>
    {chart_scores}
  </div>

  <!-- Charts Row 2: Gender + Course-wise risk -->
  <div class="charts-row" style="margin-bottom:20px;">
    <div class="chart-card">
      <div class="chart-title">👥 Gender Distribution</div>
      {chart_gender}
    </div>
    <div class="chart-card">
      <div class="chart-title">⚠️ At-Risk Students by Course</div>
      {chart_course}
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab-btn active" onclick="showTab('risk',this)">⚠️ At-Risk Students ({fail_count})</button>
    <button class="tab-btn" onclick="showTab('all',this)">📋 All Students ({total_students})</button>
  </div>

  <!-- At-Risk Table -->
  <div id="tab-risk">
    <div class="table-card">
      <div class="table-header-row">
        <div class="table-title">At-Risk Students — Complete Details</div>
        <div style="display:flex;gap:10px;align-items:center;">
          <input class="search-box" type="text" placeholder="🔍 Search student..." onkeyup="filterTable('risk-body',this.value)">
          <a href="/download/at-risk" class="dl-btn">⬇ Download CSV</a>
        </div>
      </div>
      <table>
        <thead><tr>{header_html}</tr></thead>
        <tbody id="risk-body">{risk_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- All Students Table -->
  <div id="tab-all" style="display:none;">
    <div class="table-card">
      <div class="table-header-row">
        <div class="table-title">All Students with Predicted Results</div>
        <div style="display:flex;gap:10px;align-items:center;">
          <input class="search-box" type="text" placeholder="🔍 Search student..." onkeyup="filterTable('all-body',this.value)">
          <a href="/download/all" class="dl-btn">⬇ Download CSV</a>
        </div>
      </div>
      <table>
        <thead><tr>{header_html}</tr></thead>
        <tbody id="all-body">{all_rows}</tbody>
      </table>
    </div>
  </div>

</div>

<script>
function showTab(name, btn) {{
  document.getElementById('tab-risk').style.display = name === 'risk' ? 'block' : 'none';
  document.getElementById('tab-all').style.display  = name === 'all'  ? 'block' : 'none';
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}}
function filterTable(bodyId, val) {{
  val = val.toLowerCase();
  document.querySelectorAll('#' + bodyId + ' tr').forEach(row => {{
    row.style.display = row.textContent.toLowerCase().includes(val) ? '' : 'none';
  }});
}}
</script>
</body></html>"""


# ROUTE 6: Student Dashboard


@app.route("/dashboard/student/<student_id>")
def student_dashboard(student_id):
    try:
        s = df_final[df_final["Student_ID"] == student_id].iloc[0]
    except IndexError:
        last_id = 1000 + total_students - 1
        return f"""<div style='padding:50px;font-family:Nunito,sans-serif;'>
            <h2>Student ID '{student_id}' not found.</h2>
            <p style='margin-top:10px;color:#718096'>Valid IDs: S1000 to S{last_id}</p>
            <a href='/' style='color:#667eea'>← Go back</a></div>"""

    predicted   = s["Predicted_Result"]
    performance = s.get("Performance","N/A")
    pred_perf   = s.get("Predicted_Performance","N/A")
    is_pass     = (predicted == "Pass")

    outcome_class = "outcome-pass" if is_pass else "outcome-fail"
    avatar_bg     = "#c6f6d5" if is_pass else "#fed7d7"
    avatar_color  = "#276749" if is_pass else "#9b2335"
    avatar_emoji  = "😊" if is_pass else "📚"

    student_scores = [round(float(s[c]), 1) for c in SCORE_COLS if c in df_final.columns]
    class_avgs     = [round(df_final[c].mean(), 1) for c in SCORE_COLS if c in df_final.columns]
    labels_used    = [l for l, c in zip(SCORE_LABELS, SCORE_COLS) if c in df_final.columns]

    # Grouped bar chart
    fig = go.Figure(data=[
        go.Bar(name="Your Score", x=labels_used, y=student_scores,
               marker_color="#667eea",
               text=[str(v) for v in student_scores], textposition="outside"),
        go.Bar(name="Class Average", x=labels_used, y=class_avgs,
               marker_color="#f6ad55", opacity=0.75,
               text=[str(v) for v in class_avgs], textposition="outside")
    ])
    fig.update_layout(barmode="group", paper_bgcolor="white", plot_bgcolor="white",
                      yaxis=dict(gridcolor="#f0f4ff", zeroline=False, title="Score"),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                      legend=dict(orientation="h", y=1.12),
                      margin=dict(t=30,b=10,l=10,r=10), height=300)
    chart = fig.to_html(full_html=False, config={"displayModeBar": False})

    # Progress bars
    bars_html = ""
    for label, score, avg in zip(labels_used, student_scores, class_avgs):
        diff = score - avg
        sign = "+" if diff >= 0 else ""
        diff_cls = "diff-up" if diff >= 0 else "diff-down"
        pct = min(score, 100)
        display_pct = max(pct, 2) if score == 0 else pct
        bar_style = f"width:{display_pct}%;background:linear-gradient(90deg,#667eea,#764ba2);"
        bars_html += f"""
        <div class="score-row">
          <div class="score-label">
            <span>{label}</span>
            <span>{score} &nbsp;<span class="{diff_cls} score-diff">{sign}{diff:.1f} vs class</span></span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="{bar_style}"></div>
          </div>
        </div>"""

    # Student info
    gender      = s.get("Gender","N/A")
    age         = s.get("Age","N/A")
    course      = s.get("Course","N/A")
    semester    = s.get("Semester","N/A")
    section     = s.get("Section","N/A")
    prev_gpa    = s.get("Previous_GPA","N/A")
    study_hrs   = s.get("Study_Hours_Per_Week","N/A")
    attend      = s.get("Attendance_Percentage","N/A")
    lib         = s.get("Library_Visits","N/A")
    online      = s.get("Online_Course_Hours","N/A")
    sports      = s.get("Sports_Activity","N/A")
    club        = s.get("Club_Membership","N/A")
    class_part  = s.get("Class_Participation","N/A")
    group_work  = s.get("Group_Work_Score","N/A")
    total_score = s.get("Total_Score","N/A")
    reg_no      = s.get("Registration_No","N/A")

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Student {student_id} — CS619</title>
  <style>{CSS}</style>
</head>
<body>

<div class="navbar">
  <div class="navbar-title">📊 StudentPredictor <span>Student View</span></div>
  <a href="/" class="logout-link">Logout</a>
</div>

<div class="page">

  <!-- Student Banner -->
  <div class="student-banner">
    <div class="avatar-circle" style="background:{avatar_bg};color:{avatar_color};">
      {avatar_emoji}
    </div>
    <div>
      <div class="banner-name">Student ID: {student_id} &nbsp;|&nbsp; Reg: {reg_no}</div>
      <div class="banner-sub">
        {gender} · Age {age} · Course: {course} · Semester {semester} · Section {section}<br>
        GPA: {prev_gpa} · Attendance: {attend}% · Study: {study_hrs} hrs/week · Total Score: {total_score}
      </div>
      <div class="outcome-tag {outcome_class}">
        {"✓ Predicted to PASS — Keep it up!" if is_pass else "⚠ Predicted to FAIL — Study harder!"}
      </div>
    </div>
    <div style="margin-left:auto;text-align:right;flex-shrink:0;">
      <div style="font-size:0.75rem;opacity:0.75;margin-bottom:3px;">Performance Category</div>
      <div style="font-size:1.1rem;font-weight:900;">{pred_perf}</div>
      <div style="font-size:0.75rem;opacity:0.75;margin-top:4px;">Fail Risk</div>
      <div style="font-size:1.1rem;font-weight:900;">{"Low" if is_pass else "High"}</div>
    </div>
  </div>

  <!-- Mini Stats -->
  <div class="mini-stats">
    <div class="mini-card">
      <div class="mini-val">{prev_gpa}</div>
      <div class="mini-lbl">Previous GPA</div>
    </div>
    <div class="mini-card">
      <div class="mini-val">{attend}%</div>
      <div class="mini-lbl">Attendance</div>
    </div>
    <div class="mini-card">
      <div class="mini-val">{study_hrs}</div>
      <div class="mini-lbl">Study Hrs/Wk</div>
    </div>
    <div class="mini-card">
      <div class="mini-val">{lib}</div>
      <div class="mini-lbl">Library Visits</div>
    </div>
    <div class="mini-card">
      <div class="mini-val">{online}</div>
      <div class="mini-lbl">Online Hrs</div>
    </div>
    <div class="mini-card">
      <div class="mini-val">{total_score}</div>
      <div class="mini-lbl">Total Score</div>
    </div>
  </div>

  <!-- Score Breakdown + Bar Chart -->
  <div class="two-col">
    <div class="chart-card">
      <div class="chart-title">📊 Subject Score Breakdown</div>
      {bars_html}
      <div style="margin-top:13px;padding-top:12px;border-top:1px solid #f0f4ff;font-size:0.78rem;color:#a0aec0;">
        Sports: <b style="color:#667eea">{sports}</b>
        &nbsp;·&nbsp; Club: <b style="color:#667eea">{club}</b>
        &nbsp;·&nbsp; Participation: <b style="color:#667eea">{class_part}</b>
        &nbsp;·&nbsp; Group Work: <b style="color:#667eea">{group_work}</b>
      </div>
    </div>
    <div class="chart-card">
      <div class="chart-title">📈 Your Scores vs Class Average</div>
      {chart}
    </div>
  </div>

</div>
</body></html>"""


# Start Flask server

if __name__ == "__main__":
    import logging
    # Suppress all Flask / Werkzeug internal log lines
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(debug=False, port=5000, use_reloader=False)