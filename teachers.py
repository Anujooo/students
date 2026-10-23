"""
School ERP + AI Dashboard (Streamlit)
=====================================
Run:   streamlit run school_erp_dashboard.py
Needs: pip install streamlit pandas numpy plotly scikit-learn openpyxl reportlab "qrcode[pil]"
Data:  school_students_dataset.csv in the same folder (or upload it from the sidebar).

DATA HONESTY
- REAL (from your CSV): marks, attendance, assignments, PTM, aptitude, interests, at_risk, stream.
- DEMO (generated with a fixed seed, labelled in the UI): names, fees, payroll, inventory, assets,
  hostel, health, transport/GPS, alumni, library catalogue, front-office logs.
  Replace these with real imports (Import / Export page) before showing to a school.
"""
import io
import math
import zlib
import zipfile
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas as rl_canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False
try:
    import qrcode
    HAS_QR = True
except Exception:
    HAS_QR = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    HAS_SK = True
except Exception:
    HAS_SK = False

st.set_page_config(page_title="School ERP + AI", page_icon="🏫", layout="wide")
st.markdown(
    """<style>
    div[data-testid="stMetric"]{background:rgba(120,120,160,.08);border-radius:12px;padding:10px 14px}
    .demo-tag{background:#fff3cd;color:#664d03;padding:2px 10px;border-radius:10px;font-size:.8rem}
    .real-tag{background:#d1e7dd;color:#0f5132;padding:2px 10px;border-radius:10px;font-size:.8rem}
    </style>""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------
SCHOOL_NAME = "Demo Public School"
SUBJECTS = ["math", "science", "english", "hindi", "social_science"]
SUBJ_LABEL = {s: s.replace("_", " ").title() for s in SUBJECTS}
EXAMS = ["ut1", "ut2", "halfyearly"]
EXAM_LABEL = {"ut1": "Unit Test 1", "ut2": "Unit Test 2", "halfyearly": "Half-Yearly"}
DATA_FILE = "school_students_dataset.csv"
REQUIRED_COLS = ["student_id", "class", "section", "gender", "age"]
RISK_FEATURES = [
    "prev_year_percentage", "avg_halfyearly", "score_trend", "subjects_below_33", "attendance_pct",
    "prev_month_attendance_pct", "absences_last_30d", "late_arrivals_last_30d",
    "assignment_submission_pct", "avg_days_late", "participation_score", "library_visits_month",
    "num_activities", "ptm_attended_of_4", "commute_minutes",
]  # counsellor_sessions excluded on purpose: likely a consequence of being flagged (leakage)
STREAM_FEATURES = ["aptitude_math", "aptitude_verbal", "aptitude_logical", "aptitude_creative",
                   "interest_science", "interest_commerce", "interest_humanities",
                   "math_halfyearly", "science_halfyearly", "english_halfyearly",
                   "hindi_halfyearly", "social_science_halfyearly", "avg_halfyearly"]

GROUPS = {
    "Overview": ["Principal Dashboard", "AI Risk & Alerts", "Stream Guidance"],
    "Setup / Admin": ["Admissions & Enrollment", "Roles & Permissions", "Import / Export",
                      "Certificate Generator", "Front Office Desk", "Alumni",
                      "Promotion & Transfer", "Disciplinary Records"],
    "Academic": ["Homework", "Exams", "Lesson Planner", "Assessments", "Digital Evaluation",
                 "Library", "Study Center", "CBC Academics", "Parent-Teacher Meetings"],
    "Finance": ["Fees & Accounts", "Payroll & HR", "Inventory", "Asset Management"],
    "Operations": ["Transport & GPS", "Hostel", "Student Health Records"],
    "Attendance": ["Biometric (ZKTeco)", "QR Code", "RFID / Barcode", "Attendance Analytics"],
    "Portals": ["Parent View"],
    "Founder": ["Business Plan"],
}
ALL_PAGES = [p for g in GROUPS.values() for p in g]
ROLES = ["Super Admin", "Principal", "Teacher", "Accountant", "Front Office",
         "Librarian", "Transport In-charge", "Parent"]
DEFAULT_ACCESS = {
    "Super Admin": "*", "Principal": "*",
    "Teacher": ["Principal Dashboard", "AI Risk & Alerts", "Stream Guidance", "Homework", "Exams",
                "Lesson Planner", "Assessments", "Digital Evaluation", "Study Center",
                "CBC Academics", "Parent-Teacher Meetings", "Disciplinary Records",
                "Student Health Records", "QR Code", "RFID / Barcode", "Attendance Analytics"],
    "Accountant": ["Fees & Accounts", "Payroll & HR", "Inventory", "Asset Management",
                   "Import / Export", "Admissions & Enrollment"],
    "Front Office": ["Admissions & Enrollment", "Front Office Desk", "Certificate Generator",
                     "Alumni", "QR Code", "RFID / Barcode", "Biometric (ZKTeco)"],
    "Librarian": ["Library", "Inventory"],
    "Transport In-charge": ["Transport & GPS", "RFID / Barcode"],
    "Parent": ["Parent View"],
}


# --------------------------------------------------------------------------------------
# DATA LOADING + ENRICHMENT
# --------------------------------------------------------------------------------------
MALE = ["Aarav", "Vihaan", "Arjun", "Rohan", "Kabir", "Ishaan", "Aditya", "Yash", "Dev", "Karan",
        "Naman", "Harsh", "Mohit", "Rahul", "Siddharth", "Tanish", "Vivek", "Lakshay", "Parth", "Nitin"]
FEMALE = ["Ananya", "Diya", "Isha", "Kavya", "Myra", "Navya", "Riya", "Saanvi", "Tanvi", "Aditi",
          "Pooja", "Priya", "Sneha", "Neha", "Khushi", "Shreya", "Anjali", "Muskan", "Simran", "Nisha"]
SURN = ["Sharma", "Verma", "Yadav", "Singh", "Gupta", "Malik", "Chauhan", "Jangra", "Dahiya", "Rathee",
        "Saini", "Kumar", "Goyal", "Bansal", "Mehta", "Nain", "Punia", "Sehrawat", "Beniwal", "Ahuja"]


def grade_band(x):
    if pd.isna(x):
        return None
    for cut, g in [(90, "A1"), (80, "A2"), (70, "B1"), (60, "B2"), (50, "C1"), (40, "C2"), (33, "D")]:
        if x >= cut:
            return g
    return "E"


def _name_for(sid, gender):
    rng = np.random.default_rng(zlib.crc32(str(sid).encode()))
    first = rng.choice(MALE if str(gender).lower().startswith("m") else FEMALE)
    return f"{first} {rng.choice(SURN)}"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "name" not in df.columns:
        df["name"] = [_name_for(s, g) for s, g in zip(df["student_id"], df["gender"])]
    if "status" not in df.columns:
        df["status"] = "Active"
    for c in ["class", "age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "avg_halfyearly" in df.columns:
        df["grade"] = df["avg_halfyearly"].map(grade_band)
    return df


@st.cache_data(show_spinner=False)
def read_csv_cached(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def find_default_csv():
    for base in [Path(__file__).parent if "__file__" in globals() else Path.cwd(), Path.cwd()]:
        p = base / DATA_FILE
        if p.exists():
            return p
    return None


def load_students():
    if "students" in st.session_state:
        return
    p = find_default_csv()
    up = st.sidebar.file_uploader("Upload students CSV", type=["csv"], key="boot_upload")
    if up is not None:
        st.session_state.students = enrich(read_csv_cached(up.getvalue()))
    elif p is not None:
        st.session_state.students = enrich(read_csv_cached(p.read_bytes()))
    else:
        st.info("Put `school_students_dataset.csv` next to this script or upload it in the sidebar.")
        st.stop()


def S() -> pd.DataFrame:
    return st.session_state.students


def active() -> pd.DataFrame:
    d = S()
    return d[d["status"] == "Active"]


def label(row) -> str:
    return f"{row['student_id']} · {row['name']} (Cl {row['class']}{row['section']})"


def student_picker(key, df=None, default_label=None):
    df = active() if df is None else df
    opts = {label(r): r["student_id"] for _, r in df.iterrows()}
    if not opts:
        st.warning("No students available.")
        return None
    sel = st.selectbox("Student", list(opts.keys()), key=key)
    return opts[sel]


def class_filter(key, df=None, with_all=True):
    df = active() if df is None else df
    c1, c2 = st.columns(2)
    classes = sorted(df["class"].dropna().unique().tolist())
    cl = c1.selectbox("Class", (["All"] if with_all else []) + classes, key=f"{key}_c")
    secs = sorted(df["section"].dropna().unique().tolist())
    sec = c2.selectbox("Section", (["All"] if with_all else []) + secs, key=f"{key}_s")
    out = df
    if cl != "All":
        out = out[out["class"] == cl]
    if sec != "All":
        out = out[out["section"] == sec]
    return out


def to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, d in sheets.items():
            d.to_excel(w, sheet_name=str(name)[:31], index=False)
    return buf.getvalue()


def demo_banner(text="Demo data (generated). Replace with your school's real data."):
    st.markdown(f'<span class="demo-tag">DEMO DATA</span> {text}', unsafe_allow_html=True)


def real_banner(text="Computed from your uploaded dataset."):
    st.markdown(f'<span class="real-tag">REAL DATA</span> {text}', unsafe_allow_html=True)


def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(d))


# --------------------------------------------------------------------------------------
# DEMO STATE (seeded)
# --------------------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    if ss.get("_inited"):
        return
    rng = np.random.default_rng(42)
    d = S()
    ids = d["student_id"].tolist()
    today = dt.date.today()

    # permissions matrix
    perm = pd.DataFrame(False, index=ALL_PAGES, columns=ROLES)
    for r, pages in DEFAULT_ACCESS.items():
        perm[r] = True if pages == "*" else perm.index.isin(pages)
    ss.perm = perm

    # simple logs
    ss.disciplinary = pd.DataFrame(columns=["date", "student_id", "category", "severity", "details", "action", "reported_by"])
    cats = ["Late to class", "Uniform", "Bullying", "Phone misuse", "Fighting", "Disrespect", "Cheating"]
    for sid in rng.choice(ids, 25, replace=False):
        ss.disciplinary.loc[len(ss.disciplinary)] = [
            today - dt.timedelta(days=int(rng.integers(1, 90))), sid, rng.choice(cats),
            rng.choice(["Low", "Medium", "High"], p=[.5, .35, .15]), "Logged by class teacher",
            rng.choice(["Warning", "Parent call", "Counselling", "Detention"]), "Class Teacher"]
    ss.frontdesk = {
        "enquiries": pd.DataFrame({
            "date": [today - dt.timedelta(days=int(i)) for i in rng.integers(0, 20, 12)],
            "parent": [f"{rng.choice(MALE)} {rng.choice(SURN)}" for _ in range(12)],
            "phone": [f"98{rng.integers(10000000, 99999999)}" for _ in range(12)],
            "class_wanted": rng.choice([8, 9, 10], 12),
            "source": rng.choice(["Walk-in", "Website", "Referral", "Social media"], 12),
            "status": rng.choice(["New", "Called", "Visited", "Admitted", "Lost"], 12)}),
        "visitors": pd.DataFrame(columns=["time", "visitor", "purpose", "meeting", "out_time"]),
        "calls": pd.DataFrame(columns=["time", "caller", "type", "notes"]),
        "postal": pd.DataFrame(columns=["date", "direction", "party", "ref_no", "notes"]),
    }
    ss.alumni = pd.DataFrame({
        "name": [f"{rng.choice(MALE + FEMALE)} {rng.choice(SURN)}" for _ in range(30)],
        "batch": rng.integers(2012, 2025, 30),
        "now": rng.choice(["Engineer", "Doctor", "Civil Services", "Entrepreneur", "Teacher", "Banker", "Studying abroad"], 30),
        "city": rng.choice(["Delhi", "Gurugram", "Hisar", "Rohtak", "Bengaluru", "Pune", "Toronto"], 30),
        "donation_inr": rng.choice([0, 0, 0, 5000, 11000, 25000, 51000], 30),
        "mentor": rng.choice([True, False], 30, p=[.3, .7])})
    ss.promo_backup = None
    ss.promo_history = []
    ss.transfers = pd.DataFrame(columns=["tc_no", "student_id", "date", "reason", "destination"])
    ss.users = pd.DataFrame({"user": ["admin", "principal", "t_sharma", "acc_gupta"],
                             "role": ["Super Admin", "Principal", "Teacher", "Accountant"],
                             "active": True})
    ss.homework = pd.DataFrame(columns=["class", "section", "subject", "title", "assigned", "due", "max_marks"])
    ss.lesson_plans = []
    ss.digital_eval = pd.DataFrame(columns=["student_id", "subject", "exam", "q_marks", "total", "evaluator", "remarks"])
    ss.online_quiz = []
    ss.ptm = pd.DataFrame(columns=["slot", "student_id", "teacher", "agenda", "status"])
    ss.scan_log = pd.DataFrame(columns=["time", "student_id", "method", "device"])
    ss.cert_log = pd.DataFrame(columns=["time", "student_id", "type", "serial"])
    ss.checks = {}

    # library
    titles = ["Wings of Fire", "NCERT Science 9", "Discovery of India", "Malgudi Days", "Panchatantra",
              "Atlas of India", "Oxford Dictionary", "Math Olympiad Guide", "Chacha Chaudhary", "Hindi Vyakaran"]
    ss.library_books = pd.DataFrame({"book_id": [f"B{i:03d}" for i in range(1, 41)],
                                     "title": [titles[i % 10] for i in range(40)],
                                     "copies": rng.integers(1, 6, 40), "issued": 0})
    ss.library_books["issued"] = [int(rng.integers(0, c + 1)) for c in ss.library_books["copies"]]
    ss.library_issues = pd.DataFrame(columns=["date", "student_id", "book_id", "due", "returned"])

    # fees
    fee_map = {8: 36000, 9: 40000, 10: 44000}
    fees = pd.DataFrame({"student_id": ids})
    fees["class"] = d["class"].values
    fees["annual_fee"] = fees["class"].map(fee_map).fillna(40000).astype(int)
    frac = rng.choice([1.0, 0.75, 0.5, 0.25, 0.0], len(fees), p=[.5, .2, .15, .1, .05])
    fees["paid"] = (fees["annual_fee"] * frac).astype(int)
    fees["days_overdue"] = np.where(fees["paid"] < fees["annual_fee"], rng.integers(0, 90, len(fees)), 0)
    fees["instalments_missed"] = np.where(fees["paid"] < fees["annual_fee"], rng.integers(0, 4, len(fees)), 0)
    ss.fees = fees
    ss.fee_receipts = pd.DataFrame(columns=["time", "student_id", "amount", "mode"])

    # payroll
    roles = ["PGT", "TGT", "PRT", "Clerk", "Lab Asst", "Librarian", "Driver", "Peon"]
    base = {"PGT": 52000, "TGT": 40000, "PRT": 32000, "Clerk": 22000, "Lab Asst": 20000,
            "Librarian": 26000, "Driver": 18000, "Peon": 14000}
    staff = pd.DataFrame({"emp_id": [f"E{i:03d}" for i in range(1, 31)],
                          "name": [f"{rng.choice(MALE + FEMALE)} {rng.choice(SURN)}" for _ in range(30)],
                          "designation": rng.choice(roles, 30, p=[.1, .25, .2, .08, .07, .05, .15, .1])})
    staff["basic"] = staff["designation"].map(base)
    staff["days_present"] = rng.integers(22, 27, 30)
    staff["leaves_taken"] = 26 - staff["days_present"]
    ss.staff = staff

    # inventory / assets
    ss.inventory = pd.DataFrame({
        "item": ["Chalk box", "Whiteboard marker", "A4 paper ream", "Lab beaker", "Football", "Uniform (set)",
                 "Printer toner", "First-aid kit", "Cleaning liquid", "Exam answer sheets"],
        "qty": [120, 45, 18, 60, 12, 80, 3, 6, 25, 900],
        "reorder_level": [50, 20, 20, 25, 5, 40, 4, 3, 10, 500],
        "unit_cost": [20, 30, 280, 90, 450, 700, 4500, 600, 150, 2]})
    ss.assets = pd.DataFrame({
        "asset": ["Projector", "Desktop PC", "Generator", "School bus", "Smart board", "Lab microscope", "CCTV system", "Printer"],
        "category": ["AV", "IT", "Power", "Vehicle", "AV", "Lab", "Security", "IT"],
        "purchase_date": pd.to_datetime(["2022-04-01", "2021-07-15", "2019-03-10", "2020-06-01", "2023-01-20", "2021-11-05", "2022-09-12", "2023-08-01"]),
        "cost": [45000, 38000, 250000, 1800000, 120000, 15000, 160000, 22000],
        "life_years": [8, 5, 15, 10, 10, 12, 8, 5],
        "location": ["Room 12", "Computer Lab", "Back yard", "Garage", "Room 8", "Lab 1", "Campus", "Office"]})

    # hostel
    rooms = pd.DataFrame({"room": [f"H{i:02d}" for i in range(1, 21)], "capacity": 4})
    rooms["occupants"] = [[] for _ in range(20)]
    far = d.sort_values("commute_minutes", ascending=False).head(30)["student_id"].tolist()
    for i, sid in enumerate(far):
        rooms.at[i // 4 % 20, "occupants"] = rooms.at[i // 4 % 20, "occupants"] + [sid]
    ss.hostel = rooms

    # health
    hl = pd.DataFrame({"student_id": rng.choice(ids, 120, replace=False)})
    hl["height_cm"] = rng.normal(150, 9, 120).round(0)
    hl["weight_kg"] = rng.normal(42, 8, 120).round(1).clip(22)
    hl["vision"] = rng.choice(["6/6", "6/9", "6/12", "6/18"], 120, p=[.65, .2, .1, .05])
    hl["vaccination"] = rng.choice(["Up to date", "Due", "Unknown"], 120, p=[.8, .12, .08])
    hl["last_checkup"] = [today - dt.timedelta(days=int(x)) for x in rng.integers(10, 400, 120)]
    ss.health = hl

    # transport
    ss.depot = (28.6139, 77.2090)
    ss.gps_tick = 0
    ss["_inited"] = True


# --------------------------------------------------------------------------------------
# AI: risk model, stream model, alerts
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def risk_model(data: pd.DataFrame):
    """Out-of-fold risk probability for every student (so scores are not memorised)."""
    d = data.dropna(subset=["avg_halfyearly"]).copy()
    X = d[RISK_FEATURES].fillna(d[RISK_FEATURES].median())
    y = d["at_risk"].astype(int)
    rf = RandomForestClassifier(n_estimators=250, class_weight="balanced", min_samples_leaf=3,
                                random_state=7, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    p = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=RISK_FEATURES).sort_values(ascending=False)
    return pd.Series(p, index=d["student_id"].values, name="risk_prob"), auc, imp


@st.cache_data(show_spinner=False)
def stream_model(data: pd.DataFrame):
    d = data.copy()
    X = d[STREAM_FEATURES]
    X = X.fillna(X.median())
    lab = d["stream_chosen"].notna()
    if lab.sum() < 60:
        return None, None, None
    rf = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
    y = d.loc[lab, "stream_chosen"]
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    acc = float((cross_val_predict(rf, X[lab], y, cv=cv) == y).mean())
    rf.fit(X[lab], y)
    proba = pd.DataFrame(rf.predict_proba(X), columns=rf.classes_, index=d["student_id"].values)
    return proba, acc, y.value_counts(normalize=True).max()


def build_alerts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        a = []
        if pd.notna(r.get("attendance_pct")) and r["attendance_pct"] < 75:
            a.append(("High", "Attendance below 75%"))
        if pd.notna(r.get("attendance_pct")) and r["attendance_pct"] - r["prev_month_attendance_pct"] < -5:
            a.append(("Medium", "Attendance dropped >5 pts vs last month"))
        if pd.notna(r.get("score_trend")) and r["score_trend"] < -8:
            a.append(("High", "Marks falling sharply"))
        if pd.notna(r.get("subjects_below_33")) and r["subjects_below_33"] >= 2:
            a.append(("High", f"{int(r['subjects_below_33'])} subjects below pass mark"))
        if pd.notna(r.get("assignment_submission_pct")) and r["assignment_submission_pct"] < 60:
            a.append(("Medium", "Low assignment submission"))
        if pd.notna(r.get("ptm_attended_of_4")) and r["ptm_attended_of_4"] <= 1:
            a.append(("Low", "Parent rarely attends PTM"))
        for sev, msg in a:
            rows.append({"student_id": r["student_id"], "name": r["name"], "class": r["class"],
                         "section": r["section"], "severity": sev, "alert": msg})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# PAGES: OVERVIEW
# --------------------------------------------------------------------------------------
def page_principal_dashboard():
    st.header("Principal Dashboard")
    real_banner()
    d = filt = class_filter("pd")
    c = st.columns(5)
    c[0].metric("Students", f"{len(d):,}")
    c[1].metric("Avg half-yearly %", f"{d['avg_halfyearly'].mean():.1f}")
    c[2].metric("Avg attendance %", f"{d['attendance_pct'].mean():.1f}")
    c[3].metric("Assignment submission %", f"{d['assignment_submission_pct'].mean():.1f}")
    c[4].metric("At-risk students", int(d["at_risk"].sum()), f"{d['at_risk'].mean() * 100:.1f}% of total", delta_color="off")

    a, b = st.columns(2)
    with a:
        sub = pd.DataFrame({SUBJ_LABEL[s]: [d[f"{s}_{e}"].mean() for e in EXAMS] for s in SUBJECTS},
                           index=[EXAM_LABEL[e] for e in EXAMS]).T.reset_index(names="Subject")
        fig = px.bar(sub.melt("Subject", var_name="Exam", value_name="Avg"), x="Subject", y="Avg", color="Exam",
                     barmode="group", title="Subject performance across exams")
        st.plotly_chart(fig)
    with b:
        fig = px.histogram(d, x="attendance_pct", nbins=25, title="Attendance distribution")
        fig.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="75% minimum")
        st.plotly_chart(fig)

    a, b = st.columns(2)
    with a:
        g = d.groupby(["class", "section"]).agg(avg=("avg_halfyearly", "mean"), risk=("at_risk", "mean")).reset_index()
        g["class_sec"] = g["class"].astype(str) + g["section"]
        fig = px.bar(g, x="class_sec", y="avg", color="risk", color_continuous_scale="RdYlGn_r",
                     title="Class-section average (colour = at-risk share)")
        st.plotly_chart(fig)
    with b:
        fig = px.scatter(d, x="attendance_pct", y="avg_halfyearly", color=d["at_risk"].map({0: "OK", 1: "At risk"}),
                         opacity=.6, title="Attendance vs marks", labels={"color": "Status"})
        st.plotly_chart(fig)

    st.subheader("Context factors")
    a, b, c3 = st.columns(3)
    t = d.groupby("private_tuition")["avg_halfyearly"].mean().reset_index()
    a.plotly_chart(px.bar(t, x="private_tuition", y="avg_halfyearly", title="Marks by private tuition"))
    t = d.groupby("home_internet")["avg_halfyearly"].mean().reset_index()
    b.plotly_chart(px.bar(t, x="home_internet", y="avg_halfyearly", title="Marks by home internet"))
    t = d.groupby("parent_education")["avg_halfyearly"].mean().reindex(["Below 10th", "10th-12th", "Graduate", "Postgraduate"]).reset_index()
    c3.plotly_chart(px.bar(t, x="parent_education", y="avg_halfyearly", title="Marks by parent education"))
    st.caption("These are correlations in the dataset, not causes. Use them to target support, never to label families.")


def page_ai_risk():
    st.header("AI Risk & Smart Alerts")
    real_banner("Model trained on your dataset's at_risk flag. Scores are cross-validated (out-of-fold).")
    if not HAS_SK:
        st.error("pip install scikit-learn")
        return
    d = active()
    probs, auc, imp = risk_model(d[["student_id", "at_risk", "avg_halfyearly"] + RISK_FEATURES])
    c = st.columns(3)
    c[0].metric("Cross-validated AUC", f"{auc:.3f}")
    c[1].metric("Flagged at_risk in data", int(d["at_risk"].sum()))
    c[2].metric("High-probability (>60%)", int((probs > .6).sum()))
    if auc > .95:
        st.warning("AUC is very high: in synthetic data the at_risk label is often derived from the same columns "
                   "(marks, attendance). On real school data expect lower numbers; validate before trusting it.")
    t1, t2, t3 = st.tabs(["Risk register", "Why (drivers)", "Smart alerts + parent messages"])
    with t1:
        df = filt = class_filter("risk")
        r = df.assign(risk_prob=df["student_id"].map(probs)).dropna(subset=["risk_prob"])
        r = r.sort_values("risk_prob", ascending=False)
        thr = st.slider("Show students above risk probability", 0.0, 1.0, 0.5, 0.05)
        show = r[r["risk_prob"] >= thr][["student_id", "name", "class", "section", "avg_halfyearly", "score_trend",
                                         "attendance_pct", "assignment_submission_pct", "at_risk", "risk_prob"]]
        st.dataframe(show.style.format({"risk_prob": "{:.0%}"}))
        missed = r[(r["risk_prob"] >= .5) & (r["at_risk"] == 0)]
        st.info(f"{len(missed)} students are predicted high-risk but not yet flagged in the data: review these first.")
        st.download_button("Download register (CSV)", show.to_csv(index=False), "risk_register.csv")
    with t2:
        fig = px.bar(imp.head(10)[::-1], orientation="h", title="Top drivers of at-risk prediction")
        st.plotly_chart(fig)
        st.caption("Drivers show what the model leans on, not causes. Pair every flag with a human conversation.")
    with t3:
        al = build_alerts(class_filter("alerts"))
        if al.empty:
            st.success("No alerts for this selection.")
            return
        st.dataframe(al.sort_values("severity"))
        sev = al["severity"].value_counts().reset_index()
        st.plotly_chart(px.pie(sev, names="severity", values="count", title="Alerts by severity"))
        msgs = al.copy()
        msgs["whatsapp_message"] = msgs.apply(
            lambda r: f"Dear Parent, regarding {r['name']} (Class {r['class']}{r['section']}): {r['alert']}. "
                      f"Please meet the class teacher. - {SCHOOL_NAME}", axis=1)
        st.download_button("Download parent messages (CSV)", msgs.to_csv(index=False), "parent_messages.csv")
        st.caption("Sending needs the WhatsApp Business API / SMS gateway. This exports ready-to-send text.")


def page_stream():
    st.header("Stream Guidance (Class 9-10)")
    real_banner("Trained on students who already chose a stream. Guidance, not a verdict.")
    if not HAS_SK:
        st.error("pip install scikit-learn")
        return
    d = S()
    proba, acc, base = stream_model(d[["student_id", "stream_chosen"] + STREAM_FEATURES])
    if proba is None:
        st.warning("Not enough labelled stream_chosen rows.")
        return
    c = st.columns(2)
    c[0].metric("Cross-validated accuracy", f"{acc:.0%}")
    c[1].metric("Majority-class baseline", f"{base:.0%}")
    st.caption("If accuracy is close to the baseline, the model adds little. Show the confidence bars, not just a label.")
    df = class_filter("stream", active(), with_all=True)
    df = df[df["class"].isin([8, 9, 10])]
    sid = student_picker("stream_sid", df)
    if sid is None:
        return
    r = S().set_index("student_id").loc[sid]
    p = proba.loc[sid].sort_values(ascending=False)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.bar(p.reset_index().rename(columns={"index": "Stream", sid: "Fit"}), x="index" if "index" in p.reset_index().columns else p.reset_index().columns[0], y=p.name or sid,
                               title="Model fit by stream", labels={}))
        st.write(f"Top suggestion: **{p.index[0]}** ({p.iloc[0]:.0%}). Student chose: **{r['stream_chosen'] if pd.notna(r['stream_chosen']) else 'not yet'}**")
    with b:
        cats = ["Math", "Verbal", "Logical", "Creative"]
        vals = [r["aptitude_math"], r["aptitude_verbal"], r["aptitude_logical"], r["aptitude_creative"]]
        fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself"))
        fig.update_layout(title="Aptitude profile", polar=dict(radialaxis=dict(range=[0, 100])))
        st.plotly_chart(fig)
        st.write(f"Interest (1-10): Science **{r['interest_science']}**, Commerce **{r['interest_commerce']}**, Humanities **{r['interest_humanities']}**")


# --------------------------------------------------------------------------------------
# PAGES: SETUP / ADMIN
# --------------------------------------------------------------------------------------
def page_admissions():
    st.header("Admissions & Enrollment")
    real_banner("Enrollment counts from your data. New admissions are added to this session.")
    d = S()
    t1, t2, t3 = st.tabs(["New admission", "Enrollment strength", "Enquiry funnel + lead scorer"])
    with t1:
        with st.form("adm"):
            c = st.columns(3)
            nm = c[0].text_input("Student name")
            cl = c[1].selectbox("Class", [8, 9, 10])
            sec = c[2].selectbox("Section", ["A", "B", "C", "D"])
            c = st.columns(3)
            gen = c[0].selectbox("Gender", ["Male", "Female"])
            age = c[1].number_input("Age", 10, 20, 13)
            pe = c[2].selectbox("Parent education", ["Below 10th", "10th-12th", "Graduate", "Postgraduate"])
            docs = st.multiselect("Documents received", ["Birth certificate", "Previous marksheet", "Transfer certificate", "Aadhaar", "Photos"])
            ok = st.form_submit_button("Admit student")
        if ok:
            if not nm.strip():
                st.error("Name is required.")
            else:
                n = d[(d["class"] == cl) & (d["section"] == sec)].shape[0] + 1
                sid = f"S26-{cl}{sec}-{n:04d}"
                while sid in set(d["student_id"]):
                    n += 1
                    sid = f"S26-{cl}{sec}-{n:04d}"
                new = {c: np.nan for c in d.columns}
                new.update({"student_id": sid, "name": nm.strip(), "class": cl, "section": sec, "gender": gen,
                            "age": age, "parent_education": pe, "status": "Active", "at_risk": 0})
                st.session_state.students = pd.concat([d, pd.DataFrame([new])], ignore_index=True)
                st.success(f"Admitted {nm} as {sid}. Documents: {', '.join(docs) or 'none'}")
    with t2:
        a = active()
        g = a.groupby(["class", "section"]).size().reset_index(name="students")
        st.plotly_chart(px.bar(g, x="class", y="students", color="section", barmode="group", title="Strength by class and section"))
        st.dataframe(g.pivot(index="class", columns="section", values="students"))
        gg = a.groupby(["class", "gender"]).size().reset_index(name="n")
        st.plotly_chart(px.bar(gg, x="class", y="n", color="gender", barmode="group", title="Gender split"))
    with t3:
        demo_banner("Enquiry data is simulated.")
        enq = st.session_state.frontdesk["enquiries"]
        f = enq["status"].value_counts().reindex(["New", "Called", "Visited", "Admitted", "Lost"]).fillna(0).reset_index()
        st.plotly_chart(px.funnel(f, x="count", y="status", title="Enquiry funnel"))
        w = {"Referral": 30, "Walk-in": 20, "Website": 10, "Social media": 5}
        sc = enq.copy()
        sc["lead_score"] = sc["source"].map(w) + sc["status"].map({"New": 5, "Called": 15, "Visited": 40, "Admitted": 0, "Lost": -20})
        st.write("Lead scorer (simple rules: source quality + stage). Train a model once you have real outcomes.")
        st.dataframe(sc.sort_values("lead_score", ascending=False))


def page_roles():
    st.header("Roles & Permissions")
    me = st.session_state.get("role", "Super Admin")
    st.caption("Page-level access per role (RBAC). Only Super Admin can edit. Enforced in the sidebar of this app.")
    can_edit = me == "Super Admin"
    edited = st.data_editor(st.session_state.perm, disabled=not can_edit, key="perm_editor")
    if can_edit and st.button("Save permissions"):
        st.session_state.perm = edited
        st.success("Saved. Sidebar updates on next interaction.")
    st.subheader("Users (demo)")
    st.session_state.users = st.data_editor(
        st.session_state.users, num_rows="dynamic", disabled=not can_edit, key="users_ed",
        column_config={"role": st.column_config.SelectboxColumn(options=ROLES)})
    st.info("Production checklist: hashed passwords or SSO, audit log of who viewed which child, "
            "parent consent records and minimal data collection (DPDP Act 2023), encrypted backups.")


def page_import_export():
    st.header("Import / Export")
    t1, t2 = st.tabs(["Import students", "Export"])
    with t1:
        template = pd.DataFrame(columns=REQUIRED_COLS + ["section", "attendance_pct", "avg_halfyearly"])
        st.download_button("Download blank template", template.to_csv(index=False), "students_template.csv")
        up = st.file_uploader("CSV or Excel", type=["csv", "xlsx"], key="imp")
        if up:
            df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            st.write(f"{len(df):,} rows, {len(df.columns)} columns")
            errs = []
            miss = [c for c in REQUIRED_COLS if c not in df.columns]
            if miss:
                errs.append(f"Missing required columns: {miss}")
            else:
                dup = df[df.duplicated("student_id", keep=False)]
                if len(dup):
                    errs.append(f"{len(dup)} rows with duplicate student_id")
                if df["student_id"].isna().any():
                    errs.append("Blank student_id values")
                for col in ["attendance_pct", "avg_halfyearly", "assignment_submission_pct"]:
                    if col in df.columns:
                        bad = df[(pd.to_numeric(df[col], errors="coerce") < 0) | (pd.to_numeric(df[col], errors="coerce") > 100)]
                        if len(bad):
                            errs.append(f"{len(bad)} rows have {col} outside 0-100")
            if errs:
                for e in errs:
                    st.error(e)
            else:
                st.success("Validation passed.")
                st.dataframe(df.head(20))
                mode = st.radio("Mode", ["Append new student_ids only", "Replace all students"])
                if st.button("Import"):
                    new = enrich(df)
                    if mode.startswith("Replace"):
                        st.session_state.students = new
                    else:
                        cur = S()
                        st.session_state.students = pd.concat([cur, new[~new["student_id"].isin(cur["student_id"])]], ignore_index=True)
                    st.success("Imported.")
    with t2:
        d = class_filter("exp", S())
        c = st.columns(2)
        c[0].download_button("Students CSV", d.to_csv(index=False), "students.csv")
        sheets = {"students": d, "disciplinary": st.session_state.disciplinary, "fees": st.session_state.fees,
                  "staff": st.session_state.staff, "alumni": st.session_state.alumni}
        c[1].download_button("Full workbook (Excel)", to_excel_bytes(sheets), "school_data.xlsx")


def make_certificate_pdf(kind, r, serial, date_str):
    buf = io.BytesIO()
    w, h = landscape(A4)
    c = rl_canvas.Canvas(buf, pagesize=(w, h))
    c.setStrokeColorRGB(.1, .2, .5)
    c.setLineWidth(4)
    c.rect(25, 25, w - 50, h - 50)
    c.setLineWidth(1)
    c.rect(35, 35, w - 70, h - 70)
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(w / 2, h - 90, SCHOOL_NAME.upper())
    c.setFont("Helvetica", 12)
    c.drawCentredString(w / 2, h - 110, f"Certificate No: {serial}    Date: {date_str}")
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(w / 2, h - 160, kind.upper())
    c.setFont("Helvetica", 15)
    he = "his" if str(r["gender"]).lower().startswith("m") else "her"
    body = {
        "Bonafide Certificate": f"This is to certify that {r['name']} (ID {r['student_id']}) is a bonafide student of this school, studying in Class {r['class']}{r['section']} in the current academic session.",
        "Character Certificate": f"This is to certify that {r['name']} (ID {r['student_id']}) of Class {r['class']}{r['section']} bears a good moral character and {he} conduct has been satisfactory during {he} stay at this school.",
        "School Leaving / Transfer Certificate": f"This is to certify that {r['name']} (ID {r['student_id']}) last studied in Class {r['class']}{r['section']}. All dues have been cleared and {he} is permitted to leave the school.",
        "Merit Certificate": f"This certificate is proudly presented to {r['name']} (Class {r['class']}{r['section']}) for outstanding academic performance with an average of {r.get('avg_halfyearly', float('nan')):.1f}%.",
    }[kind]
    y = h - 230
    import textwrap
    for line in textwrap.wrap(body, 80):
        c.drawCentredString(w / 2, y, line)
        y -= 26
    c.line(80, 90, 260, 90)
    c.line(w - 260, 90, w - 80, 90)
    c.setFont("Helvetica", 11)
    c.drawCentredString(170, 75, "Class Teacher")
    c.drawCentredString(w - 170, 75, "Principal")
    c.showPage()
    c.save()
    return buf.getvalue()


def page_certificates():
    st.header("Certificate Generator")
    sid = student_picker("cert_sid", S())
    if sid is None:
        return
    kinds = ["Bonafide Certificate", "Character Certificate", "School Leaving / Transfer Certificate", "Merit Certificate"]
    kind = st.selectbox("Certificate type", kinds)
    r = S().set_index("student_id").loc[sid].to_dict() | {"student_id": sid}
    n = len(st.session_state.cert_log) + 1
    serial = f"{SCHOOL_NAME[:3].upper()}/{dt.date.today().year}/{n:04d}"
    if not HAS_REPORTLAB:
        st.error('pip install reportlab (needed for PDF output)')
        return
    pdf = make_certificate_pdf(kind, r, serial, dt.date.today().strftime("%d-%m-%Y"))
    if st.download_button("Generate & download PDF", pdf, f"{kind.split()[0]}_{sid}.pdf", "application/pdf"):
        st.session_state.cert_log.loc[len(st.session_state.cert_log)] = [dt.datetime.now(), sid, kind, serial]
    st.subheader("Issue log")
    st.dataframe(st.session_state.cert_log)
    if st.checkbox("Bulk: merit certificates for a class (avg >= 85)"):
        cl = st.selectbox("Class", [8, 9, 10], key="bulkcl")
        top = active()[(active()["class"] == cl) & (active()["avg_halfyearly"] >= 85)]
        st.write(f"{len(top)} eligible")
        if len(top):
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w") as z:
                for i, (_, rr) in enumerate(top.iterrows()):
                    z.writestr(f"merit_{rr['student_id']}.pdf",
                               make_certificate_pdf("Merit Certificate", rr.to_dict(), f"BULK/{i + 1:04d}", dt.date.today().strftime("%d-%m-%Y")))
            st.download_button("Download ZIP", zbuf.getvalue(), f"merit_class{cl}.zip")


def page_frontdesk():
    st.header("Front Office Desk")
    demo_banner("Logs start with sample enquiries; entries you add stay in this session.")
    fd = st.session_state.frontdesk
    t = st.tabs(["Enquiries", "Visitors", "Call log", "Postal dispatch"])
    for tab, key in zip(t, ["enquiries", "visitors", "calls", "postal"]):
        with tab:
            fd[key] = st.data_editor(fd[key], num_rows="dynamic", key=f"fd_{key}")
    st.download_button("Export front-desk logs (Excel)", to_excel_bytes(fd), "front_office.xlsx")


def page_alumni():
    st.header("Alumni")
    demo_banner()
    al = st.session_state.alumni
    c = st.columns(3)
    c[0].metric("Alumni", len(al))
    c[1].metric("Donations (INR)", f"{int(al['donation_inr'].sum()):,}")
    c[2].metric("Willing mentors", int(al["mentor"].sum()))
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(al, x="batch", title="Alumni by batch"))
    b.plotly_chart(px.pie(al, names="now", title="Where alumni are now"))
    st.session_state.alumni = st.data_editor(al, num_rows="dynamic", key="alumni_ed")
    st.dataframe(al[al["mentor"]][["name", "batch", "now", "city"]], hide_index=True)


def page_promotion():
    st.header("Promotion & Transfer")
    real_banner("Uses your marks, attendance and subject-failure columns.")
    t1, t2 = st.tabs(["Promotion engine", "Transfer / TC"])
    with t1:
        c = st.columns(3)
        min_avg = c[0].number_input("Min average %", 0, 100, 33)
        max_fail = c[1].number_input("Max subjects below 33", 0, 5, 1)
        min_att = c[2].number_input("Min attendance %", 0, 100, 75)
        a = active()
        a = a.assign(eligible=(a["avg_halfyearly"] >= min_avg) & (a["subjects_below_33"] <= max_fail) & (a["attendance_pct"] >= min_att))
        g = a.groupby("class")["eligible"].agg(["sum", "count"]).reset_index()
        g["detained"] = g["count"] - g["sum"]
        st.dataframe(g.rename(columns={"sum": "promote", "count": "total"}))
        st.write("Students not eligible (review before detaining):")
        st.dataframe(a[~a["eligible"]][["student_id", "name", "class", "section", "avg_halfyearly", "subjects_below_33", "attendance_pct"]])
        st.warning("Applying will move eligible students up one class; Class 10 eligible students become Alumni (status 'Passed out'). You can undo once.")
        if st.button("Apply promotion"):
            st.session_state.promo_backup = S().copy()
            d = S().copy()
            ids = a[a["eligible"]]["student_id"]
            m = d["student_id"].isin(ids)
            tenth = m & (d["class"] == 10)
            d.loc[m & (d["class"] < 10), "class"] += 1
            d.loc[tenth, "status"] = "Passed out"
            st.session_state.students = d
            if tenth.sum():
                new_al = pd.DataFrame({"name": d.loc[tenth, "name"], "batch": dt.date.today().year, "now": "Class 10 pass-out",
                                       "city": "", "donation_inr": 0, "mentor": False})
                st.session_state.alumni = pd.concat([st.session_state.alumni, new_al], ignore_index=True)
            st.session_state.promo_history.append((dt.datetime.now(), int(m.sum())))
            st.success(f"Promoted {int(m.sum())} students.")
        if st.session_state.promo_backup is not None and st.button("Undo last promotion"):
            st.session_state.students = st.session_state.promo_backup
            st.session_state.promo_backup = None
            st.success("Reverted.")
        if st.session_state.promo_history:
            st.write("History:", st.session_state.promo_history)
    with t2:
        sid = student_picker("tc_sid")
        if sid:
            reason = st.selectbox("Reason", ["Parent relocation", "Change of school", "Admission elsewhere", "Other"])
            dest = st.text_input("Destination school")
            fee_due = st.session_state.fees.set_index("student_id").loc[sid]
            due = int(fee_due["annual_fee"] - fee_due["paid"]) if sid in st.session_state.fees["student_id"].values else 0
            st.write(f"Fee dues (demo data): INR {due:,}")
            if st.button("Issue TC and mark transferred", disabled=due > 0):
                n = len(st.session_state.transfers) + 1
                st.session_state.transfers.loc[n] = [f"TC/{dt.date.today().year}/{n:04d}", sid, dt.date.today(), reason, dest]
                st.session_state.students.loc[st.session_state.students["student_id"] == sid, "status"] = "Transferred"
                st.success("TC issued. Generate the PDF from Certificate Generator.")
            if due > 0:
                st.caption("TC is blocked until dues are cleared (common school policy; change if yours differs).")
        st.dataframe(st.session_state.transfers)


def page_disciplinary():
    st.header("Disciplinary Records")
    demo_banner("Sample incidents; the late-arrival and absence signals below are real.")
    ss = st.session_state
    with st.form("disc"):
        sid = student_picker("disc_sid")
        c = st.columns(3)
        cat = c[0].selectbox("Category", ["Late to class", "Uniform", "Bullying", "Phone misuse", "Fighting", "Disrespect", "Cheating", "Other"])
        sev = c[1].selectbox("Severity", ["Low", "Medium", "High"])
        act = c[2].selectbox("Action", ["Warning", "Parent call", "Counselling", "Detention", "Suspension"])
        det = st.text_area("Details")
        ok = st.form_submit_button("Log incident")
    if ok and sid:
        ss.disciplinary.loc[len(ss.disciplinary)] = [dt.date.today(), sid, cat, sev, det, act, st.session_state.get("role")]
        st.success("Logged.")
    d = ss.disciplinary
    a, b = st.columns(2)
    if len(d):
        a.plotly_chart(px.histogram(d, x="category", color="severity", title="Incidents by category"))
        rep = d["student_id"].value_counts().reset_index()
        rep = rep[rep["count"] >= 2]
        b.write("Repeat incidents (2+)")
        b.dataframe(rep)
    st.dataframe(d)
    st.subheader("Early signals from attendance data")
    sig = active()[(active()["late_arrivals_last_30d"] >= 5) | (active()["absences_last_30d"] >= 7)]
    st.dataframe(sig[["student_id", "name", "class", "section", "late_arrivals_last_30d", "absences_last_30d"]])
    st.caption("Treat these as prompts for a conversation, not as misconduct.")


# --------------------------------------------------------------------------------------
# PAGES: ACADEMIC
# --------------------------------------------------------------------------------------
def page_homework():
    st.header("Homework")
    real_banner("Submission rates are real; assigned homework list is created in-session.")
    ss = st.session_state
    t1, t2 = st.tabs(["Assign", "Submission analytics"])
    with t1:
        with st.form("hw"):
            c = st.columns(4)
            cl = c[0].selectbox("Class", [8, 9, 10])
            sec = c[1].selectbox("Section", ["A", "B", "C", "D", "All"])
            sub = c[2].selectbox("Subject", [SUBJ_LABEL[s] for s in SUBJECTS])
            mm = c[3].number_input("Max marks", 5, 100, 10)
            title = st.text_input("Title")
            due = st.date_input("Due date", dt.date.today() + dt.timedelta(days=3))
            ok = st.form_submit_button("Assign")
        if ok and title:
            ss.homework.loc[len(ss.homework)] = [cl, sec, sub, title, dt.date.today(), due, mm]
        st.dataframe(ss.homework)
    with t2:
        d = class_filter("hw")
        c = st.columns(3)
        c[0].metric("Avg submission %", f"{d['assignment_submission_pct'].mean():.1f}")
        c[1].metric("Avg days late", f"{d['avg_days_late'].mean():.2f}")
        c[2].metric("Below 60% submission", int((d["assignment_submission_pct"] < 60).sum()))
        st.plotly_chart(px.box(d, x="section", y="assignment_submission_pct", color="class", title="Submission by section"))
        st.plotly_chart(px.scatter(d, x="assignment_submission_pct", y="avg_halfyearly", trendline="ols" if _has_statsmodels() else None,
                                   title="Submission vs marks", opacity=.5))
        st.write("Students to nudge (low submission):")
        st.dataframe(d[d["assignment_submission_pct"] < 60][["student_id", "name", "class", "section", "assignment_submission_pct", "avg_days_late"]])


def _has_statsmodels():
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


def page_exams():
    st.header("Exams (online + offline)")
    ss = st.session_state
    t1, t2, t3 = st.tabs(["Results analytics", "Offline mark entry", "Online test"])
    with t1:
        real_banner()
        d = class_filter("ex")
        subj = st.selectbox("Subject", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s])
        tr = pd.DataFrame({"Exam": [EXAM_LABEL[e] for e in EXAMS], "Average": [d[f"{subj}_{e}"].mean() for e in EXAMS],
                           "Pass %": [(d[f"{subj}_{e}"] >= 33).mean() * 100 for e in EXAMS]})
        c = st.columns(2)
        c[0].plotly_chart(px.line(tr, x="Exam", y="Average", markers=True, title=f"{SUBJ_LABEL[subj]} trend"))
        c[1].plotly_chart(px.histogram(d, x=f"{subj}_halfyearly", nbins=20, title="Half-yearly score distribution"))
        top = d.nlargest(10, "avg_halfyearly")[["student_id", "name", "class", "section", "avg_halfyearly"]]
        st.write("Top 10")
        st.dataframe(top)
    with t2:
        d = class_filter("mk", with_all=False)
        subj = st.selectbox("Subject ", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s], key="mk_s")
        ex = st.selectbox("Exam", EXAMS, format_func=lambda e: EXAM_LABEL[e], key="mk_e")
        col = f"{subj}_{ex}"
        grid = d[["student_id", "name", col]].copy()
        ed = st.data_editor(grid, disabled=["student_id", "name"], key=f"grid_{col}_{len(d)}",
                            column_config={col: st.column_config.NumberColumn(min_value=0, max_value=100)})
        if st.button("Save marks"):
            S_ = S().set_index("student_id")
            S_.loc[ed["student_id"], col] = ed[col].values
            ss.students = S_.reset_index()[S().columns]
            st.success("Saved. Averages elsewhere will use these values (avg_halfyearly column is not auto-recomputed).")
    with t3:
        st.write("Build a quick MCQ test and take it.")
        with st.form("q", clear_on_submit=True):
            q = st.text_input("Question")
            opts = [st.text_input(f"Option {i + 1}") for i in range(4)]
            ans = st.selectbox("Correct option", [1, 2, 3, 4])
            if st.form_submit_button("Add question") and q and all(opts):
                ss.online_quiz.append({"q": q, "opts": opts, "ans": ans - 1})
        if ss.online_quiz:
            picks = []
            for i, it in enumerate(ss.online_quiz):
                picks.append(st.radio(f"{i + 1}. {it['q']}", it["opts"], index=None, key=f"quiz_{i}"))
            if st.button("Submit test"):
                sc = sum(1 for p, it in zip(picks, ss.online_quiz) if p == it["opts"][it["ans"]])
                st.success(f"Score: {sc}/{len(ss.online_quiz)}")
            if st.button("Clear test"):
                ss.online_quiz = []
                st.rerun()


def page_lesson_planner():
    st.header("Lesson Planner")
    st.caption("Template-based generator (no API key needed). Swap `generate_plan` for a Gemini/Claude call when you add an API key.")
    d = active()
    c = st.columns(4)
    cl = c[0].selectbox("Class", [8, 9, 10])
    subj = c[1].selectbox("Subject", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s])
    topic = c[2].text_input("Topic", "Introduction")
    mins = c[3].selectbox("Period length (min)", [30, 40, 45])
    cls = d[d["class"] == cl]
    weak = (cls[f"{subj}_halfyearly"] < 40).mean() * 100
    st.info(f"Real signal: {weak:.0f}% of Class {cl} scored below 40 in {SUBJ_LABEL[subj]} (half-yearly). "
            f"{'Include a remedial warm-up.' if weak > 15 else 'Standard pacing is fine.'}")
    if st.button("Generate plan"):
        a, b, c3, e = int(mins * .15), int(mins * .3), int(mins * .35), int(mins * .2)
        plan = f"""# {SUBJ_LABEL[subj]} · Class {cl} · {topic}
**Duration:** {mins} min

1. **Hook / recall ({a} min):** 2 quick questions from last lesson{'; add a remedial warm-up for weaker students' if weak > 15 else ''}.
2. **Teach ({b} min):** Explain {topic} using one real-life example and a board diagram.
3. **Guided practice ({c3} min):** Pairs solve 3 graded problems (easy, medium, challenge).
4. **Exit check ({e} min):** 3-question quiz; collect slips to find who needs follow-up.

**Homework:** 5 questions (2 easy, 2 medium, 1 challenge).
**Differentiation:** Peer buddy for students below 40; extension task for students above 80."""
        st.session_state.lesson_plans.append(plan)
    for p in reversed(st.session_state.lesson_plans):
        st.markdown(p)
        st.divider()
    if st.session_state.lesson_plans:
        st.download_button("Download plans (Markdown)", "\n\n".join(st.session_state.lesson_plans), "lesson_plans.md")


def page_assessments():
    st.header("Assessments")
    real_banner()
    d = class_filter("as")
    g = d["grade"].value_counts().reindex(["A1", "A2", "B1", "B2", "C1", "C2", "D", "E"]).fillna(0).reset_index()
    st.plotly_chart(px.bar(g, x="grade", y="count", title="Grade distribution (illustrative 8-band scale)"))
    fail = pd.DataFrame({SUBJ_LABEL[s]: [(d[f"{s}_halfyearly"] < 33).sum()] for s in SUBJECTS}).T.reset_index()
    fail.columns = ["Subject", "Below pass mark"]
    c = st.columns(2)
    c[0].plotly_chart(px.bar(fail, x="Subject", y="Below pass mark", title="Fails by subject (half-yearly)"))
    cm = d[[f"{s}_halfyearly" for s in SUBJECTS]].rename(columns=lambda x: SUBJ_LABEL[x.replace("_halfyearly", "")]).corr()
    c[1].plotly_chart(px.imshow(cm, text_auto=".2f", title="Subject correlation"))
    st.subheader("Improvement / decline")
    dd = d.assign(change=d["score_trend"])
    c = st.columns(2)
    c[0].write("Biggest improvers")
    c[0].dataframe(dd.nlargest(8, "change")[["student_id", "name", "class", "section", "change"]], hide_index=True)
    c[1].write("Biggest decliners")
    c[1].dataframe(dd.nsmallest(8, "change")[["student_id", "name", "class", "section", "change"]], hide_index=True)


def page_digital_eval():
    st.header("Digital Evaluation")
    st.caption("Evaluators enter question-wise marks; totals and remarks are stored with an audit trail. (Scan upload/annotation needs a PDF viewer component; add later.)")
    sid = student_picker("de_sid")
    if sid is None:
        return
    c = st.columns(3)
    subj = c[0].selectbox("Subject", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s])
    ex = c[1].selectbox("Exam", EXAMS, format_func=lambda e: EXAM_LABEL[e])
    nq = c[2].number_input("Questions", 3, 20, 6)
    scheme = pd.DataFrame({"question": [f"Q{i + 1}" for i in range(int(nq))], "max": 10, "awarded": 0.0})
    ed = st.data_editor(scheme, key=f"de_{sid}_{subj}_{ex}_{nq}", disabled=["question"])
    over = ed[ed["awarded"] > ed["max"]]
    total = float(ed["awarded"].sum())
    mx = float(ed["max"].sum())
    st.metric("Total", f"{total:.1f} / {mx:.0f}")
    rec = S().set_index("student_id").loc[sid, f"{subj}_{ex}"]
    if pd.notna(rec) and mx > 0:
        pct = total / mx * 100
        st.write(f"Recorded mark in system: **{rec}**, digital evaluation: **{pct:.0f}%**")
        if abs(pct - rec) > 10:
            st.warning("Mismatch above 10 points: send for re-evaluation.")
    rem = st.text_input("Remarks")
    if st.button("Save evaluation"):
        if len(over):
            st.error("Awarded marks exceed max for some questions.")
        else:
            st.session_state.digital_eval.loc[len(st.session_state.digital_eval)] = [sid, subj, ex, ed["awarded"].tolist(), total, st.session_state.get("role"), rem]
            st.success("Saved.")
    st.dataframe(st.session_state.digital_eval)


def page_library():
    st.header("Library")
    ss = st.session_state
    t1, t2 = st.tabs(["Catalogue & issue", "Reading vs performance (real)"])
    with t1:
        demo_banner("Catalogue is simulated.")
        bk = ss.library_books
        c = st.columns(3)
        c[0].metric("Titles", len(bk))
        c[1].metric("Copies", int(bk["copies"].sum()))
        c[2].metric("Issued", int(bk["issued"].sum()))
        with st.form("issue"):
            sid = student_picker("lib_sid")
            book = st.selectbox("Book", bk["book_id"] + " · " + bk["title"])
            ok = st.form_submit_button("Issue (14 days)")
        if ok and sid:
            bid = book.split(" · ")[0]
            row = bk[bk["book_id"] == bid].iloc[0]
            if row["issued"] >= row["copies"]:
                st.error("No copies available.")
            else:
                ss.library_books.loc[bk["book_id"] == bid, "issued"] += 1
                ss.library_issues.loc[len(ss.library_issues)] = [dt.date.today(), sid, bid, dt.date.today() + dt.timedelta(days=14), False]
                st.success("Issued.")
        st.dataframe(ss.library_issues)
        st.dataframe(ss.library_books)
    with t2:
        real_banner()
        d = active()
        d = d.assign(visits_band=pd.cut(d["library_visits_month"], [-1, 0, 2, 5, 20], labels=["0", "1-2", "3-5", "6+"]))
        g = d.groupby("visits_band", observed=True)["avg_halfyearly"].mean().reset_index()
        st.plotly_chart(px.bar(g, x="visits_band", y="avg_halfyearly", title="Marks by monthly library visits"))
        st.caption("Correlation only: motivated students may simply visit more.")


def page_study_center():
    st.header("Study Center (remedial batches)")
    real_banner("Groups are built from half-yearly marks, attendance and assignments.")
    d = class_filter("sc", with_all=True)
    c = st.columns(3)
    subj = c[0].selectbox("Subject", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s])
    cut = c[1].slider("Weak if below", 20, 60, 40)
    size = c[2].number_input("Batch size", 5, 40, 15)
    weak = d[d[f"{subj}_halfyearly"] < cut].sort_values(f"{subj}_halfyearly")
    st.write(f"{len(weak)} students need support in {SUBJ_LABEL[subj]}")
    weak = weak.assign(batch=[f"Batch {i // int(size) + 1}" for i in range(len(weak))],
                       attendance_flag=np.where(weak["attendance_pct"] < 80, "Low attendance", ""),
                       internet=weak["home_internet"])
    st.dataframe(weak[["batch", "student_id", "name", "class", "section", f"{subj}_halfyearly", "attendance_flag", "internet"]])
    nonet = int((weak["internet"] == "No").sum())
    if nonet:
        st.info(f"{nonet} of these students have no home internet: give printed material, not online-only resources.")
    st.download_button("Download batch list", weak.to_csv(index=False), f"study_center_{subj}.csv")
    if len(weak):
        st.plotly_chart(px.bar(weak["batch"].value_counts().sort_index().reset_index(), x="batch", y="count", title="Batch sizes"))


def competency(score):
    if pd.isna(score):
        return None
    return "Beginner" if score < 40 else "Progressing" if score < 60 else "Proficient" if score < 80 else "Advanced"


def page_cbc():
    st.header("CBC / Competency-Based Academics")
    real_banner("Holistic profile mixes marks with aptitude, participation, assignments and activities.")
    sid = student_picker("cbc_sid")
    if sid is None:
        return
    r = S().set_index("student_id").loc[sid]
    dims = {
        "Subject knowledge": r["avg_halfyearly"],
        "Participation": (r["participation_score"] or 0) * 20,
        "Work habits": r["assignment_submission_pct"],
        "Reasoning": r["aptitude_logical"],
        "Creativity": r["aptitude_creative"],
        "Communication": r["aptitude_verbal"],
        "Co-curricular": min(100, r["num_activities"] * 33),
    }
    vals = [0 if pd.isna(v) else v for v in dims.values()]
    c = st.columns(2)
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=list(dims) + [list(dims)[0]], fill="toself"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), title=f"{r['name']}: competency radar")
    c[0].plotly_chart(fig)
    c[1].dataframe(pd.DataFrame({"Competency": list(dims), "Score": vals, "Level": [competency(v) for v in vals]}), hide_index=True)
    st.write(f"Activity: **{r['extracurricular']}**, counsellor sessions: **{int(r['counsellor_sessions'])}**")
    st.caption("Competency bands are illustrative. Map to your board's rubric (NEP/NCF) before report cards.")
    st.subheader("Class view: competency levels")
    d = class_filter("cbcc")
    lv = d["avg_halfyearly"].map(competency).value_counts().reindex(["Beginner", "Progressing", "Proficient", "Advanced"]).fillna(0).reset_index()
    st.plotly_chart(px.bar(lv, x="avg_halfyearly", y="count", title="Subject knowledge levels"))


def page_ptm():
    st.header("Parent-Teacher Meetings")
    real_banner("Attendance at past PTMs (out of 4) is real; slots booked here are session-only.")
    d = class_filter("ptm")
    c = st.columns(3)
    c[0].metric("Avg PTMs attended / 4", f"{d['ptm_attended_of_4'].mean():.2f}")
    c[1].metric("Never attended", int((d["ptm_attended_of_4"] == 0).sum()))
    c[2].metric("Attended all 4", int((d["ptm_attended_of_4"] == 4).sum()))
    g = d.groupby("ptm_attended_of_4")["avg_halfyearly"].mean().reset_index()
    st.plotly_chart(px.bar(g, x="ptm_attended_of_4", y="avg_halfyearly", title="Marks by PTM attendance"))
    st.subheader("Priority invites (low PTM attendance and at risk)")
    pr = d[(d["ptm_attended_of_4"] <= 1) & (d["at_risk"] == 1)]
    st.dataframe(pr[["student_id", "name", "class", "section", "ptm_attended_of_4", "avg_halfyearly"]])
    st.subheader("Book a slot")
    with st.form("ptmf"):
        sid = student_picker("ptm_sid", d)
        slot = st.selectbox("Slot", [f"{h}:00" for h in range(9, 16)] + [f"{h}:30" for h in range(9, 16)])
        day = st.date_input("Date", dt.date.today() + dt.timedelta(days=7))
        ag = st.text_input("Agenda", "Progress review")
        ok = st.form_submit_button("Book")
    if ok and sid:
        key = f"{day} {slot}"
        if key in set(st.session_state.ptm["slot"]):
            st.error("Slot already booked.")
        else:
            st.session_state.ptm.loc[len(st.session_state.ptm)] = [key, sid, "Class Teacher", ag, "Confirmed"]
    st.dataframe(st.session_state.ptm)


# --------------------------------------------------------------------------------------
# PAGES: FINANCE
# --------------------------------------------------------------------------------------
def page_fees():
    st.header("Fees & Accounts")
    demo_banner("Fee amounts/payments simulated; link a real payment gateway (Razorpay/PayTM) later.")
    ss = st.session_state
    f = ss.fees.merge(S()[["student_id", "name", "section", "status"]], on="student_id")
    f = f[f["status"] == "Active"]
    f["due"] = f["annual_fee"] - f["paid"]
    c = st.columns(4)
    c[0].metric("Billed", f"₹{f['annual_fee'].sum():,}")
    c[1].metric("Collected", f"₹{f['paid'].sum():,}")
    c[2].metric("Collection %", f"{f['paid'].sum() / f['annual_fee'].sum() * 100:.1f}%")
    c[3].metric("Outstanding", f"₹{f['due'].sum():,}")
    t1, t2, t3 = st.tabs(["Collect fee", "Defaulters + predictor", "Reports"])
    with t1:
        with st.form("fee"):
            sid = student_picker("fee_sid")
            amt = st.number_input("Amount (INR)", 100, 100000, 5000, 100)
            mode = st.selectbox("Mode", ["UPI", "Cash", "Card", "Cheque", "Net banking"])
            ok = st.form_submit_button("Record payment")
        if ok and sid:
            i = ss.fees.index[ss.fees["student_id"] == sid][0]
            ss.fees.at[i, "paid"] = min(ss.fees.at[i, "annual_fee"], ss.fees.at[i, "paid"] + amt)
            ss.fee_receipts.loc[len(ss.fee_receipts)] = [dt.datetime.now(), sid, amt, mode]
            st.success("Recorded.")
        st.dataframe(ss.fee_receipts)
    with t2:
        f["risk_score"] = (0.5 * (f["due"] / f["annual_fee"]) + 0.3 * (f["days_overdue"] / 90) + 0.2 * (f["instalments_missed"] / 3)).clip(0, 1)
        top = f[f["due"] > 0].sort_values("risk_score", ascending=False).head(50)
        st.dataframe(top[["student_id", "name", "class", "due", "days_overdue", "instalments_missed", "risk_score"]].style.format({"risk_score": "{:.0%}"}))
        st.caption("Rule-based score from payment behaviour only (not family background). Train a model once you have real payment history.")
        top["reminder"] = top.apply(lambda r: f"Dear Parent, fee of Rs {r['due']:,} for {r['name']} is pending. Kindly pay at the earliest. - {SCHOOL_NAME}", axis=1)
        st.download_button("Reminder messages (CSV)", top.to_csv(index=False), "fee_reminders.csv")
    with t3:
        g = f.groupby("class")[["annual_fee", "paid", "due"]].sum().reset_index()
        st.plotly_chart(px.bar(g.melt("class"), x="class", y="value", color="variable", barmode="group", title="Fees by class"))
        st.download_button("Fee ledger (Excel)", to_excel_bytes({"ledger": f}), "fee_ledger.xlsx")


def page_payroll():
    st.header("Payroll & HR")
    demo_banner("Staff list and salary rules are placeholders. Edit percentages to match your policy.")
    c = st.columns(4)
    hra = c[0].slider("HRA % of basic", 0, 40, 20)
    da = c[1].slider("DA % of basic", 0, 30, 10)
    pf = c[2].slider("PF % of basic (employee)", 0, 15, 12)
    pt = c[3].number_input("Professional tax", 0, 500, 200)
    st.session_state.staff = st.data_editor(st.session_state.staff, num_rows="dynamic", key="staff_ed")
    s = st.session_state.staff.copy()
    s["gross"] = (s["basic"] * (1 + hra / 100 + da / 100)).round(0)
    s["lop"] = (s["gross"] / 26 * s["leaves_taken"].clip(lower=0)).round(0)
    s["pf"] = (s["basic"] * pf / 100).round(0)
    s["net"] = s["gross"] - s["lop"] - s["pf"] - pt
    st.metric("Monthly payroll (net)", f"₹{int(s['net'].sum()):,}")
    st.dataframe(s)
    st.plotly_chart(px.bar(s.groupby("designation")["net"].sum().reset_index(), x="designation", y="net", title="Payroll by designation"))
    who = st.selectbox("Payslip for", s["emp_id"] + " · " + s["name"])
    r = s[s["emp_id"] == who.split(" · ")[0]].iloc[0]
    slip = (f"PAYSLIP {dt.date.today():%B %Y}\n{SCHOOL_NAME}\n{r['name']} ({r['emp_id']}, {r['designation']})\n"
            f"Gross: {r['gross']:.0f}\nLOP: -{r['lop']:.0f}\nPF: -{r['pf']:.0f}\nProf. tax: -{pt}\nNET PAY: {r['net']:.0f}\n")
    st.code(slip)
    st.download_button("Download payslip (txt)", slip, f"payslip_{r['emp_id']}.txt")
    st.download_button("Payroll register (Excel)", to_excel_bytes({"payroll": s}), "payroll.xlsx")


def page_inventory():
    st.header("Inventory")
    demo_banner()
    ss = st.session_state
    ss.inventory = st.data_editor(ss.inventory, num_rows="dynamic", key="inv_ed")
    inv = ss.inventory.copy()
    inv["value"] = inv["qty"] * inv["unit_cost"]
    low = inv[inv["qty"] <= inv["reorder_level"]]
    c = st.columns(2)
    c[0].metric("Stock value", f"₹{int(inv['value'].sum()):,}")
    c[1].metric("Items to reorder", len(low))
    if len(low):
        st.warning("Reorder now:")
        st.dataframe(low)
    st.plotly_chart(px.bar(inv, x="item", y="value", title="Stock value by item"))


def page_assets():
    st.header("Asset Management")
    demo_banner()
    a = st.session_state.assets.copy()
    age = (pd.Timestamp.today() - a["purchase_date"]).dt.days / 365.25
    a["age_years"] = age.round(1)
    a["book_value"] = (a["cost"] * (1 - (age / a["life_years"]).clip(0, 1))).round(0)
    a["status"] = np.where(age >= a["life_years"], "Fully depreciated", "In use")
    c = st.columns(3)
    c[0].metric("Cost", f"₹{int(a['cost'].sum()):,}")
    c[1].metric("Book value", f"₹{int(a['book_value'].sum()):,}")
    c[2].metric("Replace soon", int((age >= a["life_years"] * .85).sum()))
    st.dataframe(a)
    st.plotly_chart(px.bar(a, x="asset", y=["cost", "book_value"], barmode="group", title="Cost vs book value (straight-line)"))
    with st.expander("Add asset"):
        with st.form("addasset"):
            c = st.columns(3)
            nm = c[0].text_input("Asset")
            cat = c[1].text_input("Category")
            cost = c[2].number_input("Cost", 0, 10_000_000, 10000)
            life = st.number_input("Life (years)", 1, 30, 5)
            loc = st.text_input("Location")
            if st.form_submit_button("Add") and nm:
                st.session_state.assets.loc[len(st.session_state.assets)] = [nm, cat, pd.Timestamp.today(), cost, life, loc]
                st.rerun()


# --------------------------------------------------------------------------------------
# PAGES: OPERATIONS
# --------------------------------------------------------------------------------------
def build_routes():
    ss = st.session_state
    if "routes" in ss:
        return
    rng = np.random.default_rng(7)
    depot = ss.depot
    d = active()
    riders = d[d["commute_minutes"] > 8]["student_id"].tolist()
    routes = {}
    n_buses = 8
    for b in range(n_buses):
        ang = 2 * math.pi * b / n_buses
        stops = []
        for k in range(6, 0, -1):
            rad = 0.012 * k + rng.normal(0, .002)
            jitter = rng.normal(0, .12)
            stops.append((depot[0] + rad * math.sin(ang + jitter), depot[1] + rad * math.cos(ang + jitter)))
        routes[f"Bus {b + 1}"] = {"stops": stops + [depot], "students": []}
    for i, sid in enumerate(riders):
        routes[f"Bus {i % n_buses + 1}"]["students"].append(sid)
    ss.routes = routes


def nearest_neighbour(depot, stops):
    left = stops[:]
    path, cur = [], depot
    while left:
        nxt = min(left, key=lambda s: haversine(cur, s))
        path.append(nxt)
        left.remove(nxt)
        cur = nxt
    return path


def path_len(pts):
    return sum(haversine(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def page_transport():
    st.header("Transport & GPS")
    demo_banner("Routes and bus positions are simulated. Real tracking needs a GPS device/driver app posting to an API.")
    build_routes()
    ss = st.session_state
    t1, t2, t3 = st.tabs(["Live map", "Routes & riders", "Route optimizer"])
    with t1:
        auto = st.toggle("Auto-update (3 s)", value=False)

        def panel():
            if auto:
                ss.gps_tick += 1
            elif st.button("Advance one tick"):
                ss.gps_tick += 1
            rows = []
            for name, r in ss.routes.items():
                pts = r["stops"]
                t = (ss.gps_tick * 0.25) % (len(pts) - 1)
                i = int(t)
                f = t - i
                lat = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f
                lon = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f
                rows.append({"lat": lat, "lon": lon, "size": 140, "color": "#e63946", "label": name})
                for s in pts[:-1]:
                    rows.append({"lat": s[0], "lon": s[1], "size": 30, "color": "#457b9d", "label": "stop"})
            rows.append({"lat": ss.depot[0], "lon": ss.depot[1], "size": 220, "color": "#2a9d8f", "label": "School"})
            m = pd.DataFrame(rows)
            st.map(m, latitude="lat", longitude="lon", size="size", color="color")
            st.caption("Red = buses, blue = stops, green = school. Map centre is a placeholder; change `ss.depot` to your campus.")
        if auto and hasattr(st, "fragment"):
            st.fragment(run_every=3)(panel)()
        else:
            panel()
    with t2:
        tab = pd.DataFrame({"Bus": list(ss.routes), "Stops": [len(r["stops"]) - 1 for r in ss.routes.values()],
                            "Students": [len(r["students"]) for r in ss.routes.values()]})
        tab["Seat load (cap 45)"] = (tab["Students"] / 45 * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(tab)
        st.plotly_chart(px.bar(tab, x="Bus", y="Students", title="Riders per bus (assigned by simulation)"))
        sel = st.selectbox("Bus", list(ss.routes))
        sub = S()[S()["student_id"].isin(ss.routes[sel]["students"])]
        st.dataframe(sub[["student_id", "name", "class", "section", "commute_minutes"]])
        st.caption("Riders = students whose commute is over 8 min in your data; assignment to buses is simulated.")
    with t3:
        sel = st.selectbox("Route", list(ss.routes), key="opt_bus")
        stops = ss.routes[sel]["stops"][:-1]
        shuffled = stops[:]
        np.random.default_rng(1).shuffle(shuffled)
        cur = path_len(shuffled + [ss.depot])
        opt = path_len(nearest_neighbour(ss.depot, shuffled) + [ss.depot])
        c = st.columns(3)
        c[0].metric("Unplanned order", f"{cur:.1f} km")
        c[1].metric("Optimised (nearest neighbour)", f"{opt:.1f} km")
        c[2].metric("Saving", f"{(1 - opt / cur) * 100:.0f}%")
        st.caption("Greedy heuristic: fast and decent. Use OR-Tools for capacity/time-window constraints.")


def page_hostel():
    st.header("Hostel")
    demo_banner("Rooms are simulated. Students with the longest commute in your data are seeded as residents.")
    h = st.session_state.hostel
    occ = h["occupants"].map(len)
    c = st.columns(3)
    c[0].metric("Beds", int(h["capacity"].sum()))
    c[1].metric("Occupied", int(occ.sum()))
    c[2].metric("Vacant", int(h["capacity"].sum() - occ.sum()))
    view = h.assign(occupied=occ, occupants=h["occupants"].map(lambda x: ", ".join(x)))
    st.plotly_chart(px.bar(view, x="room", y="occupied", title="Occupancy by room"))
    st.dataframe(view)
    st.subheader("Allocate")
    with st.form("alloc"):
        sid = student_picker("host_sid")
        free = [r for r, o, cap in zip(h["room"], occ, h["capacity"]) if o < cap]
        room = st.selectbox("Room", free)
        ok = st.form_submit_button("Allocate")
    if ok and sid:
        if any(sid in o for o in h["occupants"]):
            st.error("Already allocated.")
        else:
            i = h.index[h["room"] == room][0]
            st.session_state.hostel.at[i, "occupants"] = h.at[i, "occupants"] + [sid]
            st.rerun()
    far = active().nlargest(10, "commute_minutes")[["student_id", "name", "class", "commute_minutes"]]
    st.write("Suggested candidates (longest commute, real data):")
    st.dataframe(far)


def page_health():
    st.header("Student Health Records")
    demo_banner("Sample records. Health data is sensitive: restrict to nurse/principal and log access (DPDP Act).")
    h = st.session_state.health.copy()
    h["bmi"] = (h["weight_kg"] / (h["height_cm"] / 100) ** 2).round(1)
    h["bmi_band"] = pd.cut(h["bmi"], [0, 14.5, 22, 27, 100], labels=["Low", "Normal", "High", "Very high"])
    c = st.columns(3)
    c[0].metric("Records", len(h))
    c[1].metric("Vaccination due/unknown", int((h["vaccination"] != "Up to date").sum()))
    c[2].metric("Vision below 6/9", int(h["vision"].isin(["6/12", "6/18"]).sum()))
    a, b = st.columns(2)
    a.plotly_chart(px.pie(h, names="vision", title="Vision screening"))
    b.plotly_chart(px.histogram(h, x="bmi_band", title="BMI bands (screening only, age-adjusted charts needed)"))
    st.dataframe(h.merge(S()[["student_id", "name", "class", "section"]], on="student_id"))
    with st.expander("Add / update record"):
        with st.form("hl"):
            sid = student_picker("hl_sid")
            c = st.columns(3)
            ht = c[0].number_input("Height (cm)", 90, 200, 150)
            wt = c[1].number_input("Weight (kg)", 15.0, 120.0, 42.0)
            vis = c[2].selectbox("Vision", ["6/6", "6/9", "6/12", "6/18"])
            vac = st.selectbox("Vaccination", ["Up to date", "Due", "Unknown"])
            if st.form_submit_button("Save") and sid:
                hh = st.session_state.health
                hh = hh[hh["student_id"] != sid]
                hh.loc[len(hh)] = [sid, ht, wt, vis, vac, dt.date.today()]
                st.session_state.health = hh
                st.rerun()


# --------------------------------------------------------------------------------------
# PAGES: ATTENDANCE
# --------------------------------------------------------------------------------------
def add_scan(code, method, device):
    d = S()
    code = code.strip()
    if code not in set(d["student_id"]):
        return False, "Unknown ID"
    log = st.session_state.scan_log
    today = dt.date.today()
    already = ((log["student_id"] == code) & (pd.to_datetime(log["time"]).dt.date == today)).any() if len(log) else False
    st.session_state.scan_log.loc[len(log)] = [dt.datetime.now(), code, method, device]
    return True, ("Already marked today (logged as repeat punch)" if already else "Marked present")


def scan_panel(method, device, help_text):
    st.caption(help_text)
    code = st.text_input("Scan / type ID and press Enter", key=f"scan_{method}")
    if code:
        ok, msg = add_scan(code, method, device)
        (st.success if ok else st.error)(f"{code}: {msg}")
    log = st.session_state.scan_log
    mine = log[log["method"] == method]
    c = st.columns(2)
    c[0].metric("Scans today", int(len(mine[pd.to_datetime(mine["time"]).dt.date == dt.date.today()])) if len(mine) else 0)
    c[1].metric("Unique students today", int(mine["student_id"].nunique()) if len(mine) else 0)
    st.dataframe(mine.tail(20))


def page_biometric():
    st.header("Biometric (ZKTeco)")
    st.caption("ZKTeco devices push punches to an ADMS endpoint or export an attendance log (.dat/.csv). "
               "Streamlit cannot receive device pushes directly: run a small FastAPI receiver that writes to your DB, "
               "or import the exported log here.")
    t1, t2 = st.tabs(["Import device log", "Simulate punches"])
    with t1:
        up = st.file_uploader("ZKTeco export (CSV/TXT: user_id, datetime, verify, status)", type=["csv", "txt", "dat"])
        if up:
            raw = pd.read_csv(up, sep=None, engine="python", header=None)
            raw = raw.iloc[:, :4]
            raw.columns = ["user_id", "datetime", "verify", "status"][:raw.shape[1]]
            raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
            raw["user_id"] = raw["user_id"].astype(str).str.strip()
            ok = raw[raw["user_id"].isin(set(S()["student_id"])) & raw["datetime"].notna()]
            st.write(f"{len(ok)} of {len(raw)} rows matched a student_id (device PIN must equal student_id).")
            st.dataframe(raw.head(20))
            if st.button("Add matched punches to log") and len(ok):
                add = pd.DataFrame({"time": ok["datetime"], "student_id": ok["user_id"], "method": "Biometric", "device": "ZKTeco"})
                st.session_state.scan_log = pd.concat([st.session_state.scan_log, add], ignore_index=True)
                st.success("Added.")
    with t2:
        n = st.slider("Simulate morning punches", 50, 800, 300, 50)
        if st.button("Run simulation"):
            rng = np.random.default_rng()
            d = active().sample(min(n, len(active())), random_state=None)
            base = dt.datetime.combine(dt.date.today(), dt.time(7, 30))
            add = pd.DataFrame({"time": [base + dt.timedelta(minutes=int(m)) for m in rng.integers(0, 60, len(d))],
                                "student_id": d["student_id"].values, "method": "Biometric", "device": "ZKTeco"})
            st.session_state.scan_log = pd.concat([st.session_state.scan_log, add], ignore_index=True)
            st.success(f"{len(d)} punches simulated.")
    mine = st.session_state.scan_log
    st.dataframe(mine[mine["method"] == "Biometric"].tail(20))


def page_qr():
    st.header("QR Code Attendance")
    t1, t2 = st.tabs(["Scan", "Generate ID cards"])
    with t1:
        scan_panel("QR", "Phone camera or USB QR scanner", "A USB/Bluetooth scanner types the code like a keyboard: click the box and scan.")
    with t2:
        if not HAS_QR:
            st.error('pip install "qrcode[pil]"')
            return
        d = class_filter("qrc", with_all=False)
        sid = student_picker("qr_sid", d)
        if sid:
            img = qrcode.make(sid)
            b = io.BytesIO()
            img.save(b, format="PNG")
            st.image(b.getvalue(), width=180, caption=sid)
        if st.button("Build ZIP of QR codes for this class/section"):
            zb = io.BytesIO()
            with zipfile.ZipFile(zb, "w") as z:
                for s in d["student_id"]:
                    bb = io.BytesIO()
                    qrcode.make(s).save(bb, format="PNG")
                    z.writestr(f"{s}.png", bb.getvalue())
            st.download_button("Download ZIP", zb.getvalue(), "qr_codes.zip")


def page_rfid():
    st.header("RFID / Barcode Attendance")
    scan_panel("RFID/Barcode", "Gate reader", "RFID readers and barcode scanners act like keyboards. Program each card/barcode with the student_id.")
    st.caption("Reuse the same reader at the bus door to mark boarding. Add a second log table keyed by bus.")


def page_att_analytics():
    st.header("Attendance Analytics")
    real_banner("Month-level attendance from your dataset, plus today's scans from this session.")
    d = class_filter("aa")
    c = st.columns(4)
    c[0].metric("Avg attendance %", f"{d['attendance_pct'].mean():.1f}")
    c[1].metric("Below 75%", int((d["attendance_pct"] < 75).sum()))
    c[2].metric("Dropped >5 pts vs last month", int(((d["attendance_pct"] - d["prev_month_attendance_pct"]) < -5).sum()))
    c[3].metric("Avg late arrivals / 30d", f"{d['late_arrivals_last_30d'].mean():.1f}")
    a, b = st.columns(2)
    g = d.groupby(["class", "section"])[["prev_month_attendance_pct", "attendance_pct"]].mean().reset_index()
    g["cs"] = g["class"].astype(str) + g["section"]
    a.plotly_chart(px.bar(g.melt(["class", "section", "cs"]), x="cs", y="value", color="variable", barmode="group", title="Previous vs current attendance"))
    b.plotly_chart(px.scatter(d, x="attendance_pct", y="avg_halfyearly", color="at_risk", title="Attendance vs marks", opacity=.5))
    st.subheader("Chronic absentees")
    ch = d[d["attendance_pct"] < 75].sort_values("attendance_pct")
    st.dataframe(ch[["student_id", "name", "class", "section", "attendance_pct", "absences_last_30d"]])
    log = st.session_state.scan_log
    if len(log):
        t = log[pd.to_datetime(log["time"]).dt.date == dt.date.today()]
        st.subheader("Today (from scans)")
        c = st.columns(3)
        c[0].metric("Present", int(t["student_id"].nunique()))
        c[1].metric("Absent", int(len(active()) - t["student_id"].nunique()))
        c[2].metric("Present %", f"{t['student_id'].nunique() / len(active()) * 100:.1f}")
        by = t.groupby("method")["student_id"].nunique().reset_index()
        st.plotly_chart(px.bar(by, x="method", y="student_id", title="Unique students by method"))


# --------------------------------------------------------------------------------------
# PAGES: PORTALS + FOUNDER
# --------------------------------------------------------------------------------------
def page_parent_view():
    st.header("Parent View")
    st.caption("What a parent app would show. In production, a parent can see only their own child (login-linked).")
    sid = student_picker("pv_sid")
    if sid is None:
        return
    r = S().set_index("student_id").loc[sid]
    c = st.columns(4)
    c[0].metric("Average %", f"{r['avg_halfyearly']:.1f}" if pd.notna(r["avg_halfyearly"]) else "-")
    c[1].metric("Attendance %", f"{r['attendance_pct']:.1f}" if pd.notna(r["attendance_pct"]) else "-")
    c[2].metric("Assignments %", f"{r['assignment_submission_pct']:.0f}" if pd.notna(r["assignment_submission_pct"]) else "-")
    f = st.session_state.fees
    due = int(f[f["student_id"] == sid].eval("annual_fee - paid").iloc[0]) if sid in set(f["student_id"]) else 0
    c[3].metric("Fee due (demo)", f"₹{due:,}")
    if pd.notna(r["avg_halfyearly"]):
        tr = pd.DataFrame([{"Subject": SUBJ_LABEL[s], "Exam": EXAM_LABEL[e], "Marks": r[f"{s}_{e}"]} for s in SUBJECTS for e in EXAMS])
        st.plotly_chart(px.line(tr, x="Exam", y="Marks", color="Subject", markers=True, title="Marks across exams"))
        al = build_alerts(S()[S()["student_id"] == sid])
        if len(al):
            st.warning("Things to discuss with the teacher: " + "; ".join(al["alert"]))
        else:
            st.success("No concerns flagged.")


def page_business():
    st.header("Business Plan: School SaaS")
    st.caption("Your plan from this chat: sell to Indian schools at about ₹100 per student per year.")
    t1, t2, t3 = st.tabs(["Unit economics", "Competitor benchmark", "Roadmap"])
    with t1:
        c = st.columns(3)
        schools = c[0].number_input("Schools signed", 1, 500, 5)
        students = c[1].number_input("Avg students per school", 100, 5000, 1500, 100)
        price = c[2].number_input("Price per student / year (INR)", 20, 500, 100, 10)
        c = st.columns(3)
        cac = c[0].number_input("Acquisition cost per school (INR)", 0, 500000, 40000, 5000)
        sup = c[1].number_input("Support cost per school / month (INR)", 0, 100000, 6000, 500)
        infra = c[2].number_input("Cloud + SMS/WhatsApp per student / year (INR)", 0, 100, 12)
        rev = schools * students * price
        cost = schools * cac + schools * sup * 12 + schools * students * infra
        c = st.columns(3)
        c[0].metric("Annual revenue", f"₹{rev:,.0f}")
        c[1].metric("Annual cost (year 1)", f"₹{cost:,.0f}")
        c[2].metric("Year-1 margin", f"₹{rev - cost:,.0f}")
        ratio = rev / max(cost, 1)
        st.progress(min(1.0, ratio / 2), text=f"Revenue / cost = {ratio:.2f}x")
        sweep = pd.DataFrame({"schools": range(1, 31)})
        sweep["margin"] = sweep["schools"] * students * price - (sweep["schools"] * cac + sweep["schools"] * sup * 12 + sweep["schools"] * students * infra)
        st.plotly_chart(px.line(sweep, x="schools", y="margin", title="Year-1 margin vs number of schools"))
        st.caption("Costs ignore your own salary, taxes and payment delays (schools often pay late). Model those before committing.")
    with t2:
        comp = pd.DataFrame({"Plan": ["Starter", "Standard", "Premium"], "Monthly (INR)": [1999, 2499, 3599], "Max students": [300, 500, 1000]})
        comp["Effective INR / student / year"] = (comp["Monthly (INR)"] * 12 / comp["Max students"]).round(1)
        st.dataframe(comp)
        st.write("From the Multi School ERP pricing page you shared. At ₹100 you are above these at full capacity, so lead with the AI early-warning layer, not a bundle of 50 modules.")
        feat = pd.DataFrame({
            "Area": ["Core ERP (fees, attendance, exams)", "Biometric / GPS / RFID", "AI risk + early warning", "Stream guidance", "Explainable alerts to parents"],
            "Big ERP": ["Yes", "Yes", "Claimed, no detail", "Rare", "Rare"],
            "Your wedge": ["Basic", "Phase 2", "Phase 1 focus", "Phase 1 focus", "Phase 1 focus"]})
        st.dataframe(feat)
    with t3:
        phases = {
            "Phase 0 (wk 1-3): Validate": ["Interview 10 principals/teachers", "Pick ONE painful problem", "Get 2-3 pilot schools"],
            "Phase 1 (wk 3-10): MVP": ["CSV import + student records", "Principal dashboard", "Role-based logins", "Risk flags + alerts"],
            "Phase 2 (mo 3-5): Pilot": ["Run free pilot with real data", "WhatsApp/SMS reminders", "Measure outcomes (flags that proved right)"],
            "Phase 3 (parallel): Trust": ["DPDP consent + privacy policy", "Encrypted backups + audit log", "Data processing agreement"],
            "Phase 4 (mo 5-9): First customers": ["Convert pilots to paid", "Demo deck + 5-min video", "Support routine (WhatsApp helpdesk, SLA)"],
            "Phase 5 (mo 9-18): Scale": ["Fees/report-card modules", "Integrations with existing ERPs", "Hire support, then sales"],
        }
        done = total = 0
        for ph, items in phases.items():
            with st.expander(ph, expanded=False):
                for it in items:
                    k = f"rm_{ph}_{it}"
                    v = st.checkbox(it, key=k)
                    total += 1
                    done += int(v)
        st.progress(done / total if total else 0, text=f"{done}/{total} tasks done")
    st.caption("Do not copy competitors' claims ('2000+ schools', 'bank-grade'). Only publish numbers you can prove.")


# --------------------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------------------
PAGE_FUNCS = {
    "Principal Dashboard": page_principal_dashboard, "AI Risk & Alerts": page_ai_risk, "Stream Guidance": page_stream,
    "Admissions & Enrollment": page_admissions, "Roles & Permissions": page_roles, "Import / Export": page_import_export,
    "Certificate Generator": page_certificates, "Front Office Desk": page_frontdesk, "Alumni": page_alumni,
    "Promotion & Transfer": page_promotion, "Disciplinary Records": page_disciplinary,
    "Homework": page_homework, "Exams": page_exams, "Lesson Planner": page_lesson_planner, "Assessments": page_assessments,
    "Digital Evaluation": page_digital_eval, "Library": page_library, "Study Center": page_study_center,
    "CBC Academics": page_cbc, "Parent-Teacher Meetings": page_ptm,
    "Fees & Accounts": page_fees, "Payroll & HR": page_payroll, "Inventory": page_inventory, "Asset Management": page_assets,
    "Transport & GPS": page_transport, "Hostel": page_hostel, "Student Health Records": page_health,
    "Biometric (ZKTeco)": page_biometric, "QR Code": page_qr, "RFID / Barcode": page_rfid, "Attendance Analytics": page_att_analytics,
    "Parent View": page_parent_view, "Business Plan": page_business,
}


def main():
    st.sidebar.title(f"🏫 {SCHOOL_NAME}")
    load_students()
    init_state()
    role = st.sidebar.selectbox("Sign in as (demo)", ROLES, key="role")
    perm = st.session_state.perm
    allowed = set(perm.index[perm[role]]) if role in perm.columns else set()
    if role == "Super Admin":
        allowed = set(ALL_PAGES)
    groups = {g: [p for p in ps if p in allowed] for g, ps in GROUPS.items()}
    groups = {g: ps for g, ps in groups.items() if ps}
    if not groups:
        st.error("This role has no pages enabled.")
        return
    group = st.sidebar.selectbox("Module", list(groups), key="group")
    page = st.sidebar.radio("Page", groups[group], key=f"page_{group}")
    a = active()
    st.sidebar.divider()
    st.sidebar.caption(f"{len(a):,} active students · {a['avg_halfyearly'].notna().sum():,} with marks")
    st.sidebar.caption("Student data is personal data of minors: collect only what you need, get parental consent (DPDP Act 2023).")
    PAGE_FUNCS[page]()


main()"""
School ERP + AI Dashboard (Streamlit)
=====================================
Run:   streamlit run school_erp_dashboard.py
Needs: pip install streamlit pandas numpy plotly scikit-learn openpyxl reportlab "qrcode[pil]"
Data:  school_students_dataset.csv in the same folder (or upload it from the sidebar).

DATA HONESTY
- REAL (from your CSV): marks, attendance, assignments, PTM, aptitude, interests, at_risk, stream.
- DEMO (generated with a fixed seed, labelled in the UI): names, fees, payroll, inventory, assets,
  hostel, health, transport/GPS, alumni, library catalogue, front-office logs.
  Replace these with real imports (Import / Export page) before showing to a school.
"""
import io
import math
import zlib
import zipfile
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas as rl_canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False
try:
    import qrcode
    HAS_QR = True
except Exception:
    HAS_QR = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    HAS_SK = True
except Exception:
    HAS_SK = False

st.set_page_config(page_title="School ERP + AI", page_icon="🏫", layout="wide")
st.markdown(
    """<style>
    div[data-testid="stMetric"]{background:rgba(120,120,160,.08);border-radius:12px;padding:10px 14px}
    .demo-tag{background:#fff3cd;color:#664d03;padding:2px 10px;border-radius:10px;font-size:.8rem}
    .real-tag{background:#d1e7dd;color:#0f5132;padding:2px 10px;border-radius:10px;font-size:.8rem}
    </style>""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------
SCHOOL_NAME = "Demo Public School"
SUBJECTS = ["math", "science", "english", "hindi", "social_science"]
SUBJ_LABEL = {s: s.replace("_", " ").title() for s in SUBJECTS}
EXAMS = ["ut1", "ut2", "halfyearly"]
EXAM_LABEL = {"ut1": "Unit Test 1", "ut2": "Unit Test 2", "halfyearly": "Half-Yearly"}
DATA_FILE = "school_students_dataset.csv"
REQUIRED_COLS = ["student_id", "class", "section", "gender", "age"]
RISK_FEATURES = [
    "prev_year_percentage", "avg_halfyearly", "score_trend", "subjects_below_33", "attendance_pct",
    "prev_month_attendance_pct", "absences_last_30d", "late_arrivals_last_30d",
    "assignment_submission_pct", "avg_days_late", "participation_score", "library_visits_month",
    "num_activities", "ptm_attended_of_4", "commute_minutes",
]  # counsellor_sessions excluded on purpose: likely a consequence of being flagged (leakage)
STREAM_FEATURES = ["aptitude_math", "aptitude_verbal", "aptitude_logical", "aptitude_creative",
                   "interest_science", "interest_commerce", "interest_humanities",
                   "math_halfyearly", "science_halfyearly", "english_halfyearly",
                   "hindi_halfyearly", "social_science_halfyearly", "avg_halfyearly"]

GROUPS = {
    "Overview": ["Principal Dashboard", "AI Risk & Alerts", "Stream Guidance"],
    "Setup / Admin": ["Admissions & Enrollment", "Roles & Permissions", "Import / Export",
                      "Certificate Generator", "Front Office Desk", "Alumni",
                      "Promotion & Transfer", "Disciplinary Records"],
    "Academic": ["Homework", "Exams", "Lesson Planner", "Assessments", "Digital Evaluation",
                 "Library", "Study Center", "CBC Academics", "Parent-Teacher Meetings"],
    "Finance": ["Fees & Accounts", "Payroll & HR", "Inventory", "Asset Management"],
    "Operations": ["Transport & GPS", "Hostel", "Student Health Records"],
    "Attendance": ["Biometric (ZKTeco)", "QR Code", "RFID / Barcode", "Attendance Analytics"],
    "Portals": ["Parent View"],
    "Founder": ["Business Plan"],
}
ALL_PAGES = [p for g in GROUPS.values() for p in g]
ROLES = ["Super Admin", "Principal", "Teacher", "Accountant", "Front Office",
         "Librarian", "Transport In-charge", "Parent"]
DEFAULT_ACCESS = {
    "Super Admin": "*", "Principal": "*",
    "Teacher": ["Principal Dashboard", "AI Risk & Alerts", "Stream Guidance", "Homework", "Exams",
                "Lesson Planner", "Assessments", "Digital Evaluation", "Study Center",
                "CBC Academics", "Parent-Teacher Meetings", "Disciplinary Records",
                "Student Health Records", "QR Code", "RFID / Barcode", "Attendance Analytics"],
    "Accountant": ["Fees & Accounts", "Payroll & HR", "Inventory", "Asset Management",
                   "Import / Export", "Admissions & Enrollment"],
    "Front Office": ["Admissions & Enrollment", "Front Office Desk", "Certificate Generator",
                     "Alumni", "QR Code", "RFID / Barcode", "Biometric (ZKTeco)"],
    "Librarian": ["Library", "Inventory"],
    "Transport In-charge": ["Transport & GPS", "RFID / Barcode"],
    "Parent": ["Parent View"],
}


# --------------------------------------------------------------------------------------
# DATA LOADING + ENRICHMENT
# --------------------------------------------------------------------------------------
MALE = ["Aarav", "Vihaan", "Arjun", "Rohan", "Kabir", "Ishaan", "Aditya", "Yash", "Dev", "Karan",
        "Naman", "Harsh", "Mohit", "Rahul", "Siddharth", "Tanish", "Vivek", "Lakshay", "Parth", "Nitin"]
FEMALE = ["Ananya", "Diya", "Isha", "Kavya", "Myra", "Navya", "Riya", "Saanvi", "Tanvi", "Aditi",
          "Pooja", "Priya", "Sneha", "Neha", "Khushi", "Shreya", "Anjali", "Muskan", "Simran", "Nisha"]
SURN = ["Sharma", "Verma", "Yadav", "Singh", "Gupta", "Malik", "Chauhan", "Jangra", "Dahiya", "Rathee",
        "Saini", "Kumar", "Goyal", "Bansal", "Mehta", "Nain", "Punia", "Sehrawat", "Beniwal", "Ahuja"]


def grade_band(x):
    if pd.isna(x):
        return None
    for cut, g in [(90, "A1"), (80, "A2"), (70, "B1"), (60, "B2"), (50, "C1"), (40, "C2"), (33, "D")]:
        if x >= cut:
            return g
    return "E"


def _name_for(sid, gender):
    rng = np.random.default_rng(zlib.crc32(str(sid).encode()))
    first = rng.choice(MALE if str(gender).lower().startswith("m") else FEMALE)
    return f"{first} {rng.choice(SURN)}"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "name" not in df.columns:
        df["name"] = [_name_for(s, g) for s, g in zip(df["student_id"], df["gender"])]
    if "status" not in df.columns:
        df["status"] = "Active"
    for c in ["class", "age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "avg_halfyearly" in df.columns:
        df["grade"] = df["avg_halfyearly"].map(grade_band)
    return df


@st.cache_data(show_spinner=False)
def read_csv_cached(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def find_default_csv():
    for base in [Path(__file__).parent if "__file__" in globals() else Path.cwd(), Path.cwd()]:
        p = base / DATA_FILE
        if p.exists():
            return p
    return None


def load_students():
    if "students" in st.session_state:
        return
    p = find_default_csv()
    up = st.sidebar.file_uploader("Upload students CSV", type=["csv"], key="boot_upload")
    if up is not None:
        st.session_state.students = enrich(read_csv_cached(up.getvalue()))
    elif p is not None:
        st.session_state.students = enrich(read_csv_cached(p.read_bytes()))
    else:
        st.info("Put `school_students_dataset.csv` next to this script or upload it in the sidebar.")
        st.stop()


def S() -> pd.DataFrame:
    return st.session_state.students


def active() -> pd.DataFrame:
    d = S()
    return d[d["status"] == "Active"]


def label(row) -> str:
    return f"{row['student_id']} · {row['name']} (Cl {row['class']}{row['section']})"


def student_picker(key, df=None, default_label=None):
    df = active() if df is None else df
    opts = {label(r): r["student_id"] for _, r in df.iterrows()}
    if not opts:
        st.warning("No students available.")
        return None
    sel = st.selectbox("Student", list(opts.keys()), key=key)
    return opts[sel]


def class_filter(key, df=None, with_all=True):
    df = active() if df is None else df
    c1, c2 = st.columns(2)
    classes = sorted(df["class"].dropna().unique().tolist())
    cl = c1.selectbox("Class", (["All"] if with_all else []) + classes, key=f"{key}_c")
    secs = sorted(df["section"].dropna().unique().tolist())
    sec = c2.selectbox("Section", (["All"] if with_all else []) + secs, key=f"{key}_s")
    out = df
    if cl != "All":
        out = out[out["class"] == cl]
    if sec != "All":
        out = out[out["section"] == sec]
    return out


def to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, d in sheets.items():
            d.to_excel(w, sheet_name=str(name)[:31], index=False)
    return buf.getvalue()


def demo_banner(text="Demo data (generated). Replace with your school's real data."):
    st.markdown(f'<span class="demo-tag">DEMO DATA</span> {text}', unsafe_allow_html=True)


def real_banner(text="Computed from your uploaded dataset."):
    st.markdown(f'<span class="real-tag">REAL DATA</span> {text}', unsafe_allow_html=True)


def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(d))


# --------------------------------------------------------------------------------------
# DEMO STATE (seeded)
# --------------------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    if ss.get("_inited"):
        return
    rng = np.random.default_rng(42)
    d = S()
    ids = d["student_id"].tolist()
    today = dt.date.today()

    # permissions matrix
    perm = pd.DataFrame(False, index=ALL_PAGES, columns=ROLES)
    for r, pages in DEFAULT_ACCESS.items():
        perm[r] = True if pages == "*" else perm.index.isin(pages)
    ss.perm = perm

    # simple logs
    ss.disciplinary = pd.DataFrame(columns=["date", "student_id", "category", "severity", "details", "action", "reported_by"])
    cats = ["Late to class", "Uniform", "Bullying", "Phone misuse", "Fighting", "Disrespect", "Cheating"]
    for sid in rng.choice(ids, 25, replace=False):
        ss.disciplinary.loc[len(ss.disciplinary)] = [
            today - dt.timedelta(days=int(rng.integers(1, 90))), sid, rng.choice(cats),
            rng.choice(["Low", "Medium", "High"], p=[.5, .35, .15]), "Logged by class teacher",
            rng.choice(["Warning", "Parent call", "Counselling", "Detention"]), "Class Teacher"]
    ss.frontdesk = {
        "enquiries": pd.DataFrame({
            "date": [today - dt.timedelta(days=int(i)) for i in rng.integers(0, 20, 12)],
            "parent": [f"{rng.choice(MALE)} {rng.choice(SURN)}" for _ in range(12)],
            "phone": [f"98{rng.integers(10000000, 99999999)}" for _ in range(12)],
            "class_wanted": rng.choice([8, 9, 10], 12),
            "source": rng.choice(["Walk-in", "Website", "Referral", "Social media"], 12),
            "status": rng.choice(["New", "Called", "Visited", "Admitted", "Lost"], 12)}),
        "visitors": pd.DataFrame(columns=["time", "visitor", "purpose", "meeting", "out_time"]),
        "calls": pd.DataFrame(columns=["time", "caller", "type", "notes"]),
        "postal": pd.DataFrame(columns=["date", "direction", "party", "ref_no", "notes"]),
    }
    ss.alumni = pd.DataFrame({
        "name": [f"{rng.choice(MALE + FEMALE)} {rng.choice(SURN)}" for _ in range(30)],
        "batch": rng.integers(2012, 2025, 30),
        "now": rng.choice(["Engineer", "Doctor", "Civil Services", "Entrepreneur", "Teacher", "Banker", "Studying abroad"], 30),
        "city": rng.choice(["Delhi", "Gurugram", "Hisar", "Rohtak", "Bengaluru", "Pune", "Toronto"], 30),
        "donation_inr": rng.choice([0, 0, 0, 5000, 11000, 25000, 51000], 30),
        "mentor": rng.choice([True, False], 30, p=[.3, .7])})
    ss.promo_backup = None
    ss.promo_history = []
    ss.transfers = pd.DataFrame(columns=["tc_no", "student_id", "date", "reason", "destination"])
    ss.users = pd.DataFrame({"user": ["admin", "principal", "t_sharma", "acc_gupta"],
                             "role": ["Super Admin", "Principal", "Teacher", "Accountant"],
                             "active": True})
    ss.homework = pd.DataFrame(columns=["class", "section", "subject", "title", "assigned", "due", "max_marks"])
    ss.lesson_plans = []
    ss.digital_eval = pd.DataFrame(columns=["student_id", "subject", "exam", "q_marks", "total", "evaluator", "remarks"])
    ss.online_quiz = []
    ss.ptm = pd.DataFrame(columns=["slot", "student_id", "teacher", "agenda", "status"])
    ss.scan_log = pd.DataFrame(columns=["time", "student_id", "method", "device"])
    ss.cert_log = pd.DataFrame(columns=["time", "student_id", "type", "serial"])
    ss.checks = {}

    # library
    titles = ["Wings of Fire", "NCERT Science 9", "Discovery of India", "Malgudi Days", "Panchatantra",
              "Atlas of India", "Oxford Dictionary", "Math Olympiad Guide", "Chacha Chaudhary", "Hindi Vyakaran"]
    ss.library_books = pd.DataFrame({"book_id": [f"B{i:03d}" for i in range(1, 41)],
                                     "title": [titles[i % 10] for i in range(40)],
                                     "copies": rng.integers(1, 6, 40), "issued": 0})
    ss.library_books["issued"] = [int(rng.integers(0, c + 1)) for c in ss.library_books["copies"]]
    ss.library_issues = pd.DataFrame(columns=["date", "student_id", "book_id", "due", "returned"])

    # fees
    fee_map = {8: 36000, 9: 40000, 10: 44000}
    fees = pd.DataFrame({"student_id": ids})
    fees["class"] = d["class"].values
    fees["annual_fee"] = fees["class"].map(fee_map).fillna(40000).astype(int)
    frac = rng.choice([1.0, 0.75, 0.5, 0.25, 0.0], len(fees), p=[.5, .2, .15, .1, .05])
    fees["paid"] = (fees["annual_fee"] * frac).astype(int)
    fees["days_overdue"] = np.where(fees["paid"] < fees["annual_fee"], rng.integers(0, 90, len(fees)), 0)
    fees["instalments_missed"] = np.where(fees["paid"] < fees["annual_fee"], rng.integers(0, 4, len(fees)), 0)
    ss.fees = fees
    ss.fee_receipts = pd.DataFrame(columns=["time", "student_id", "amount", "mode"])

    # payroll
    roles = ["PGT", "TGT", "PRT", "Clerk", "Lab Asst", "Librarian", "Driver", "Peon"]
    base = {"PGT": 52000, "TGT": 40000, "PRT": 32000, "Clerk": 22000, "Lab Asst": 20000,
            "Librarian": 26000, "Driver": 18000, "Peon": 14000}
    staff = pd.DataFrame({"emp_id": [f"E{i:03d}" for i in range(1, 31)],
                          "name": [f"{rng.choice(MALE + FEMALE)} {rng.choice(SURN)}" for _ in range(30)],
                          "designation": rng.choice(roles, 30, p=[.1, .25, .2, .08, .07, .05, .15, .1])})
    staff["basic"] = staff["designation"].map(base)
    staff["days_present"] = rng.integers(22, 27, 30)
    staff["leaves_taken"] = 26 - staff["days_present"]
    ss.staff = staff

    # inventory / assets
    ss.inventory = pd.DataFrame({
        "item": ["Chalk box", "Whiteboard marker", "A4 paper ream", "Lab beaker", "Football", "Uniform (set)",
                 "Printer toner", "First-aid kit", "Cleaning liquid", "Exam answer sheets"],
        "qty": [120, 45, 18, 60, 12, 80, 3, 6, 25, 900],
        "reorder_level": [50, 20, 20, 25, 5, 40, 4, 3, 10, 500],
        "unit_cost": [20, 30, 280, 90, 450, 700, 4500, 600, 150, 2]})
    ss.assets = pd.DataFrame({
        "asset": ["Projector", "Desktop PC", "Generator", "School bus", "Smart board", "Lab microscope", "CCTV system", "Printer"],
        "category": ["AV", "IT", "Power", "Vehicle", "AV", "Lab", "Security", "IT"],
        "purchase_date": pd.to_datetime(["2022-04-01", "2021-07-15", "2019-03-10", "2020-06-01", "2023-01-20", "2021-11-05", "2022-09-12", "2023-08-01"]),
        "cost": [45000, 38000, 250000, 1800000, 120000, 15000, 160000, 22000],
        "life_years": [8, 5, 15, 10, 10, 12, 8, 5],
        "location": ["Room 12", "Computer Lab", "Back yard", "Garage", "Room 8", "Lab 1", "Campus", "Office"]})

    # hostel
    rooms = pd.DataFrame({"room": [f"H{i:02d}" for i in range(1, 21)], "capacity": 4})
    rooms["occupants"] = [[] for _ in range(20)]
    far = d.sort_values("commute_minutes", ascending=False).head(30)["student_id"].tolist()
    for i, sid in enumerate(far):
        rooms.at[i // 4 % 20, "occupants"] = rooms.at[i // 4 % 20, "occupants"] + [sid]
    ss.hostel = rooms

    # health
    hl = pd.DataFrame({"student_id": rng.choice(ids, 120, replace=False)})
    hl["height_cm"] = rng.normal(150, 9, 120).round(0)
    hl["weight_kg"] = rng.normal(42, 8, 120).round(1).clip(22)
    hl["vision"] = rng.choice(["6/6", "6/9", "6/12", "6/18"], 120, p=[.65, .2, .1, .05])
    hl["vaccination"] = rng.choice(["Up to date", "Due", "Unknown"], 120, p=[.8, .12, .08])
    hl["last_checkup"] = [today - dt.timedelta(days=int(x)) for x in rng.integers(10, 400, 120)]
    ss.health = hl

    # transport
    ss.depot = (28.6139, 77.2090)
    ss.gps_tick = 0
    ss["_inited"] = True


# --------------------------------------------------------------------------------------
# AI: risk model, stream model, alerts
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def risk_model(data: pd.DataFrame):
    """Out-of-fold risk probability for every student (so scores are not memorised)."""
    d = data.dropna(subset=["avg_halfyearly"]).copy()
    X = d[RISK_FEATURES].fillna(d[RISK_FEATURES].median())
    y = d["at_risk"].astype(int)
    rf = RandomForestClassifier(n_estimators=250, class_weight="balanced", min_samples_leaf=3,
                                random_state=7, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    p = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=RISK_FEATURES).sort_values(ascending=False)
    return pd.Series(p, index=d["student_id"].values, name="risk_prob"), auc, imp


@st.cache_data(show_spinner=False)
def stream_model(data: pd.DataFrame):
    d = data.copy()
    X = d[STREAM_FEATURES]
    X = X.fillna(X.median())
    lab = d["stream_chosen"].notna()
    if lab.sum() < 60:
        return None, None, None
    rf = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
    y = d.loc[lab, "stream_chosen"]
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    acc = float((cross_val_predict(rf, X[lab], y, cv=cv) == y).mean())
    rf.fit(X[lab], y)
    proba = pd.DataFrame(rf.predict_proba(X), columns=rf.classes_, index=d["student_id"].values)
    return proba, acc, y.value_counts(normalize=True).max()


def build_alerts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        a = []
        if pd.notna(r.get("attendance_pct")) and r["attendance_pct"] < 75:
            a.append(("High", "Attendance below 75%"))
        if pd.notna(r.get("attendance_pct")) and r["attendance_pct"] - r["prev_month_attendance_pct"] < -5:
            a.append(("Medium", "Attendance dropped >5 pts vs last month"))
        if pd.notna(r.get("score_trend")) and r["score_trend"] < -8:
            a.append(("High", "Marks falling sharply"))
        if pd.notna(r.get("subjects_below_33")) and r["subjects_below_33"] >= 2:
            a.append(("High", f"{int(r['subjects_below_33'])} subjects below pass mark"))
        if pd.notna(r.get("assignment_submission_pct")) and r["assignment_submission_pct"] < 60:
            a.append(("Medium", "Low assignment submission"))
        if pd.notna(r.get("ptm_attended_of_4")) and r["ptm_attended_of_4"] <= 1:
            a.append(("Low", "Parent rarely attends PTM"))
        for sev, msg in a:
            rows.append({"student_id": r["student_id"], "name": r["name"], "class": r["class"],
                         "section": r["section"], "severity": sev, "alert": msg})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# PAGES: OVERVIEW
# --------------------------------------------------------------------------------------
def page_principal_dashboard():
    st.header("Principal Dashboard")
    real_banner()
    d = filt = class_filter("pd")
    c = st.columns(5)
    c[0].metric("Students", f"{len(d):,}")
    c[1].metric("Avg half-yearly %", f"{d['avg_halfyearly'].mean():.1f}")
    c[2].metric("Avg attendance %", f"{d['attendance_pct'].mean():.1f}")
    c[3].metric("Assignment submission %", f"{d['assignment_submission_pct'].mean():.1f}")
    c[4].metric("At-risk students", int(d["at_risk"].sum()), f"{d['at_risk'].mean() * 100:.1f}% of total", delta_color="off")

    a, b = st.columns(2)
    with a:
        sub = pd.DataFrame({SUBJ_LABEL[s]: [d[f"{s}_{e}"].mean() for e in EXAMS] for s in SUBJECTS},
                           index=[EXAM_LABEL[e] for e in EXAMS]).T.reset_index(names="Subject")
        fig = px.bar(sub.melt("Subject", var_name="Exam", value_name="Avg"), x="Subject", y="Avg", color="Exam",
                     barmode="group", title="Subject performance across exams")
        st.plotly_chart(fig)
    with b:
        fig = px.histogram(d, x="attendance_pct", nbins=25, title="Attendance distribution")
        fig.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="75% minimum")
        st.plotly_chart(fig)

    a, b = st.columns(2)
    with a:
        g = d.groupby(["class", "section"]).agg(avg=("avg_halfyearly", "mean"), risk=("at_risk", "mean")).reset_index()
        g["class_sec"] = g["class"].astype(str) + g["section"]
        fig = px.bar(g, x="class_sec", y="avg", color="risk", color_continuous_scale="RdYlGn_r",
                     title="Class-section average (colour = at-risk share)")
        st.plotly_chart(fig)
    with b:
        fig = px.scatter(d, x="attendance_pct", y="avg_halfyearly", color=d["at_risk"].map({0: "OK", 1: "At risk"}),
                         opacity=.6, title="Attendance vs marks", labels={"color": "Status"})
        st.plotly_chart(fig)

    st.subheader("Context factors")
    a, b, c3 = st.columns(3)
    t = d.groupby("private_tuition")["avg_halfyearly"].mean().reset_index()
    a.plotly_chart(px.bar(t, x="private_tuition", y="avg_halfyearly", title="Marks by private tuition"))
    t = d.groupby("home_internet")["avg_halfyearly"].mean().reset_index()
    b.plotly_chart(px.bar(t, x="home_internet", y="avg_halfyearly", title="Marks by home internet"))
    t = d.groupby("parent_education")["avg_halfyearly"].mean().reindex(["Below 10th", "10th-12th", "Graduate", "Postgraduate"]).reset_index()
    c3.plotly_chart(px.bar(t, x="parent_education", y="avg_halfyearly", title="Marks by parent education"))
    st.caption("These are correlations in the dataset, not causes. Use them to target support, never to label families.")


def page_ai_risk():
    st.header("AI Risk & Smart Alerts")
    real_banner("Model trained on your dataset's at_risk flag. Scores are cross-validated (out-of-fold).")
    if not HAS_SK:
        st.error("pip install scikit-learn")
        return
    d = active()
    probs, auc, imp = risk_model(d[["student_id", "at_risk", "avg_halfyearly"] + RISK_FEATURES])
    c = st.columns(3)
    c[0].metric("Cross-validated AUC", f"{auc:.3f}")
    c[1].metric("Flagged at_risk in data", int(d["at_risk"].sum()))
    c[2].metric("High-probability (>60%)", int((probs > .6).sum()))
    if auc > .95:
        st.warning("AUC is very high: in synthetic data the at_risk label is often derived from the same columns "
                   "(marks, attendance). On real school data expect lower numbers; validate before trusting it.")
    t1, t2, t3 = st.tabs(["Risk register", "Why (drivers)", "Smart alerts + parent messages"])
    with t1:
        df = filt = class_filter("risk")
        r = df.assign(risk_prob=df["student_id"].map(probs)).dropna(subset=["risk_prob"])
        r = r.sort_values("risk_prob", ascending=False)
        thr = st.slider("Show students above risk probability", 0.0, 1.0, 0.5, 0.05)
        show = r[r["risk_prob"] >= thr][["student_id", "name", "class", "section", "avg_halfyearly", "score_trend",
                                         "attendance_pct", "assignment_submission_pct", "at_risk", "risk_prob"]]
        st.dataframe(show.style.format({"risk_prob": "{:.0%}"}))
        missed = r[(r["risk_prob"] >= .5) & (r["at_risk"] == 0)]
        st.info(f"{len(missed)} students are predicted high-risk but not yet flagged in the data: review these first.")
        st.download_button("Download register (CSV)", show.to_csv(index=False), "risk_register.csv")
    with t2:
        fig = px.bar(imp.head(10)[::-1], orientation="h", title="Top drivers of at-risk prediction")
        st.plotly_chart(fig)
        st.caption("Drivers show what the model leans on, not causes. Pair every flag with a human conversation.")
    with t3:
        al = build_alerts(class_filter("alerts"))
        if al.empty:
            st.success("No alerts for this selection.")
            return
        st.dataframe(al.sort_values("severity"))
        sev = al["severity"].value_counts().reset_index()
        st.plotly_chart(px.pie(sev, names="severity", values="count", title="Alerts by severity"))
        msgs = al.copy()
        msgs["whatsapp_message"] = msgs.apply(
            lambda r: f"Dear Parent, regarding {r['name']} (Class {r['class']}{r['section']}): {r['alert']}. "
                      f"Please meet the class teacher. - {SCHOOL_NAME}", axis=1)
        st.download_button("Download parent messages (CSV)", msgs.to_csv(index=False), "parent_messages.csv")
        st.caption("Sending needs the WhatsApp Business API / SMS gateway. This exports ready-to-send text.")


def page_stream():
    st.header("Stream Guidance (Class 9-10)")
    real_banner("Trained on students who already chose a stream. Guidance, not a verdict.")
    if not HAS_SK:
        st.error("pip install scikit-learn")
        return
    d = S()
    proba, acc, base = stream_model(d[["student_id", "stream_chosen"] + STREAM_FEATURES])
    if proba is None:
        st.warning("Not enough labelled stream_chosen rows.")
        return
    c = st.columns(2)
    c[0].metric("Cross-validated accuracy", f"{acc:.0%}")
    c[1].metric("Majority-class baseline", f"{base:.0%}")
    st.caption("If accuracy is close to the baseline, the model adds little. Show the confidence bars, not just a label.")
    df = class_filter("stream", active(), with_all=True)
    df = df[df["class"].isin([8, 9, 10])]
    sid = student_picker("stream_sid", df)
    if sid is None:
        return
    r = S().set_index("student_id").loc[sid]
    p = proba.loc[sid].sort_values(ascending=False)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.bar(p.reset_index().rename(columns={"index": "Stream", sid: "Fit"}), x="index" if "index" in p.reset_index().columns else p.reset_index().columns[0], y=p.name or sid,
                               title="Model fit by stream", labels={}))
        st.write(f"Top suggestion: **{p.index[0]}** ({p.iloc[0]:.0%}). Student chose: **{r['stream_chosen'] if pd.notna(r['stream_chosen']) else 'not yet'}**")
    with b:
        cats = ["Math", "Verbal", "Logical", "Creative"]
        vals = [r["aptitude_math"], r["aptitude_verbal"], r["aptitude_logical"], r["aptitude_creative"]]
        fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself"))
        fig.update_layout(title="Aptitude profile", polar=dict(radialaxis=dict(range=[0, 100])))
        st.plotly_chart(fig)
        st.write(f"Interest (1-10): Science **{r['interest_science']}**, Commerce **{r['interest_commerce']}**, Humanities **{r['interest_humanities']}**")


# --------------------------------------------------------------------------------------
# PAGES: SETUP / ADMIN
# --------------------------------------------------------------------------------------
def page_admissions():
    st.header("Admissions & Enrollment")
    real_banner("Enrollment counts from your data. New admissions are added to this session.")
    d = S()
    t1, t2, t3 = st.tabs(["New admission", "Enrollment strength", "Enquiry funnel + lead scorer"])
    with t1:
        with st.form("adm"):
            c = st.columns(3)
            nm = c[0].text_input("Student name")
            cl = c[1].selectbox("Class", [8, 9, 10])
            sec = c[2].selectbox("Section", ["A", "B", "C", "D"])
            c = st.columns(3)
            gen = c[0].selectbox("Gender", ["Male", "Female"])
            age = c[1].number_input("Age", 10, 20, 13)
            pe = c[2].selectbox("Parent education", ["Below 10th", "10th-12th", "Graduate", "Postgraduate"])
            docs = st.multiselect("Documents received", ["Birth certificate", "Previous marksheet", "Transfer certificate", "Aadhaar", "Photos"])
            ok = st.form_submit_button("Admit student")
        if ok:
            if not nm.strip():
                st.error("Name is required.")
            else:
                n = d[(d["class"] == cl) & (d["section"] == sec)].shape[0] + 1
                sid = f"S26-{cl}{sec}-{n:04d}"
                while sid in set(d["student_id"]):
                    n += 1
                    sid = f"S26-{cl}{sec}-{n:04d}"
                new = {c: np.nan for c in d.columns}
                new.update({"student_id": sid, "name": nm.strip(), "class": cl, "section": sec, "gender": gen,
                            "age": age, "parent_education": pe, "status": "Active", "at_risk": 0})
                st.session_state.students = pd.concat([d, pd.DataFrame([new])], ignore_index=True)
                st.success(f"Admitted {nm} as {sid}. Documents: {', '.join(docs) or 'none'}")
    with t2:
        a = active()
        g = a.groupby(["class", "section"]).size().reset_index(name="students")
        st.plotly_chart(px.bar(g, x="class", y="students", color="section", barmode="group", title="Strength by class and section"))
        st.dataframe(g.pivot(index="class", columns="section", values="students"))
        gg = a.groupby(["class", "gender"]).size().reset_index(name="n")
        st.plotly_chart(px.bar(gg, x="class", y="n", color="gender", barmode="group", title="Gender split"))
    with t3:
        demo_banner("Enquiry data is simulated.")
        enq = st.session_state.frontdesk["enquiries"]
        f = enq["status"].value_counts().reindex(["New", "Called", "Visited", "Admitted", "Lost"]).fillna(0).reset_index()
        st.plotly_chart(px.funnel(f, x="count", y="status", title="Enquiry funnel"))
        w = {"Referral": 30, "Walk-in": 20, "Website": 10, "Social media": 5}
        sc = enq.copy()
        sc["lead_score"] = sc["source"].map(w) + sc["status"].map({"New": 5, "Called": 15, "Visited": 40, "Admitted": 0, "Lost": -20})
        st.write("Lead scorer (simple rules: source quality + stage). Train a model once you have real outcomes.")
        st.dataframe(sc.sort_values("lead_score", ascending=False))


def page_roles():
    st.header("Roles & Permissions")
    me = st.session_state.get("role", "Super Admin")
    st.caption("Page-level access per role (RBAC). Only Super Admin can edit. Enforced in the sidebar of this app.")
    can_edit = me == "Super Admin"
    edited = st.data_editor(st.session_state.perm, disabled=not can_edit, key="perm_editor")
    if can_edit and st.button("Save permissions"):
        st.session_state.perm = edited
        st.success("Saved. Sidebar updates on next interaction.")
    st.subheader("Users (demo)")
    st.session_state.users = st.data_editor(
        st.session_state.users, num_rows="dynamic", disabled=not can_edit, key="users_ed",
        column_config={"role": st.column_config.SelectboxColumn(options=ROLES)})
    st.info("Production checklist: hashed passwords or SSO, audit log of who viewed which child, "
            "parent consent records and minimal data collection (DPDP Act 2023), encrypted backups.")


def page_import_export():
    st.header("Import / Export")
    t1, t2 = st.tabs(["Import students", "Export"])
    with t1:
        template = pd.DataFrame(columns=REQUIRED_COLS + ["section", "attendance_pct", "avg_halfyearly"])
        st.download_button("Download blank template", template.to_csv(index=False), "students_template.csv")
        up = st.file_uploader("CSV or Excel", type=["csv", "xlsx"], key="imp")
        if up:
            df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            st.write(f"{len(df):,} rows, {len(df.columns)} columns")
            errs = []
            miss = [c for c in REQUIRED_COLS if c not in df.columns]
            if miss:
                errs.append(f"Missing required columns: {miss}")
            else:
                dup = df[df.duplicated("student_id", keep=False)]
                if len(dup):
                    errs.append(f"{len(dup)} rows with duplicate student_id")
                if df["student_id"].isna().any():
                    errs.append("Blank student_id values")
                for col in ["attendance_pct", "avg_halfyearly", "assignment_submission_pct"]:
                    if col in df.columns:
                        bad = df[(pd.to_numeric(df[col], errors="coerce") < 0) | (pd.to_numeric(df[col], errors="coerce") > 100)]
                        if len(bad):
                            errs.append(f"{len(bad)} rows have {col} outside 0-100")
            if errs:
                for e in errs:
                    st.error(e)
            else:
                st.success("Validation passed.")
                st.dataframe(df.head(20))
                mode = st.radio("Mode", ["Append new student_ids only", "Replace all students"])
                if st.button("Import"):
                    new = enrich(df)
                    if mode.startswith("Replace"):
                        st.session_state.students = new
                    else:
                        cur = S()
                        st.session_state.students = pd.concat([cur, new[~new["student_id"].isin(cur["student_id"])]], ignore_index=True)
                    st.success("Imported.")
    with t2:
        d = class_filter("exp", S())
        c = st.columns(2)
        c[0].download_button("Students CSV", d.to_csv(index=False), "students.csv")
        sheets = {"students": d, "disciplinary": st.session_state.disciplinary, "fees": st.session_state.fees,
                  "staff": st.session_state.staff, "alumni": st.session_state.alumni}
        c[1].download_button("Full workbook (Excel)", to_excel_bytes(sheets), "school_data.xlsx")


def make_certificate_pdf(kind, r, serial, date_str):
    buf = io.BytesIO()
    w, h = landscape(A4)
    c = rl_canvas.Canvas(buf, pagesize=(w, h))
    c.setStrokeColorRGB(.1, .2, .5)
    c.setLineWidth(4)
    c.rect(25, 25, w - 50, h - 50)
    c.setLineWidth(1)
    c.rect(35, 35, w - 70, h - 70)
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(w / 2, h - 90, SCHOOL_NAME.upper())
    c.setFont("Helvetica", 12)
    c.drawCentredString(w / 2, h - 110, f"Certificate No: {serial}    Date: {date_str}")
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(w / 2, h - 160, kind.upper())
    c.setFont("Helvetica", 15)
    he = "his" if str(r["gender"]).lower().startswith("m") else "her"
    body = {
        "Bonafide Certificate": f"This is to certify that {r['name']} (ID {r['student_id']}) is a bonafide student of this school, studying in Class {r['class']}{r['section']} in the current academic session.",
        "Character Certificate": f"This is to certify that {r['name']} (ID {r['student_id']}) of Class {r['class']}{r['section']} bears a good moral character and {he} conduct has been satisfactory during {he} stay at this school.",
        "School Leaving / Transfer Certificate": f"This is to certify that {r['name']} (ID {r['student_id']}) last studied in Class {r['class']}{r['section']}. All dues have been cleared and {he} is permitted to leave the school.",
        "Merit Certificate": f"This certificate is proudly presented to {r['name']} (Class {r['class']}{r['section']}) for outstanding academic performance with an average of {r.get('avg_halfyearly', float('nan')):.1f}%.",
    }[kind]
    y = h - 230
    import textwrap
    for line in textwrap.wrap(body, 80):
        c.drawCentredString(w / 2, y, line)
        y -= 26
    c.line(80, 90, 260, 90)
    c.line(w - 260, 90, w - 80, 90)
    c.setFont("Helvetica", 11)
    c.drawCentredString(170, 75, "Class Teacher")
    c.drawCentredString(w - 170, 75, "Principal")
    c.showPage()
    c.save()
    return buf.getvalue()


def page_certificates():
    st.header("Certificate Generator")
    sid = student_picker("cert_sid", S())
    if sid is None:
        return
    kinds = ["Bonafide Certificate", "Character Certificate", "School Leaving / Transfer Certificate", "Merit Certificate"]
    kind = st.selectbox("Certificate type", kinds)
    r = S().set_index("student_id").loc[sid].to_dict() | {"student_id": sid}
    n = len(st.session_state.cert_log) + 1
    serial = f"{SCHOOL_NAME[:3].upper()}/{dt.date.today().year}/{n:04d}"
    if not HAS_REPORTLAB:
        st.error('pip install reportlab (needed for PDF output)')
        return
    pdf = make_certificate_pdf(kind, r, serial, dt.date.today().strftime("%d-%m-%Y"))
    if st.download_button("Generate & download PDF", pdf, f"{kind.split()[0]}_{sid}.pdf", "application/pdf"):
        st.session_state.cert_log.loc[len(st.session_state.cert_log)] = [dt.datetime.now(), sid, kind, serial]
    st.subheader("Issue log")
    st.dataframe(st.session_state.cert_log)
    if st.checkbox("Bulk: merit certificates for a class (avg >= 85)"):
        cl = st.selectbox("Class", [8, 9, 10], key="bulkcl")
        top = active()[(active()["class"] == cl) & (active()["avg_halfyearly"] >= 85)]
        st.write(f"{len(top)} eligible")
        if len(top):
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w") as z:
                for i, (_, rr) in enumerate(top.iterrows()):
                    z.writestr(f"merit_{rr['student_id']}.pdf",
                               make_certificate_pdf("Merit Certificate", rr.to_dict(), f"BULK/{i + 1:04d}", dt.date.today().strftime("%d-%m-%Y")))
            st.download_button("Download ZIP", zbuf.getvalue(), f"merit_class{cl}.zip")


def page_frontdesk():
    st.header("Front Office Desk")
    demo_banner("Logs start with sample enquiries; entries you add stay in this session.")
    fd = st.session_state.frontdesk
    t = st.tabs(["Enquiries", "Visitors", "Call log", "Postal dispatch"])
    for tab, key in zip(t, ["enquiries", "visitors", "calls", "postal"]):
        with tab:
            fd[key] = st.data_editor(fd[key], num_rows="dynamic", key=f"fd_{key}")
    st.download_button("Export front-desk logs (Excel)", to_excel_bytes(fd), "front_office.xlsx")


def page_alumni():
    st.header("Alumni")
    demo_banner()
    al = st.session_state.alumni
    c = st.columns(3)
    c[0].metric("Alumni", len(al))
    c[1].metric("Donations (INR)", f"{int(al['donation_inr'].sum()):,}")
    c[2].metric("Willing mentors", int(al["mentor"].sum()))
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(al, x="batch", title="Alumni by batch"))
    b.plotly_chart(px.pie(al, names="now", title="Where alumni are now"))
    st.session_state.alumni = st.data_editor(al, num_rows="dynamic", key="alumni_ed")
    st.dataframe(al[al["mentor"]][["name", "batch", "now", "city"]], hide_index=True)


def page_promotion():
    st.header("Promotion & Transfer")
    real_banner("Uses your marks, attendance and subject-failure columns.")
    t1, t2 = st.tabs(["Promotion engine", "Transfer / TC"])
    with t1:
        c = st.columns(3)
        min_avg = c[0].number_input("Min average %", 0, 100, 33)
        max_fail = c[1].number_input("Max subjects below 33", 0, 5, 1)
        min_att = c[2].number_input("Min attendance %", 0, 100, 75)
        a = active()
        a = a.assign(eligible=(a["avg_halfyearly"] >= min_avg) & (a["subjects_below_33"] <= max_fail) & (a["attendance_pct"] >= min_att))
        g = a.groupby("class")["eligible"].agg(["sum", "count"]).reset_index()
        g["detained"] = g["count"] - g["sum"]
        st.dataframe(g.rename(columns={"sum": "promote", "count": "total"}))
        st.write("Students not eligible (review before detaining):")
        st.dataframe(a[~a["eligible"]][["student_id", "name", "class", "section", "avg_halfyearly", "subjects_below_33", "attendance_pct"]])
        st.warning("Applying will move eligible students up one class; Class 10 eligible students become Alumni (status 'Passed out'). You can undo once.")
        if st.button("Apply promotion"):
            st.session_state.promo_backup = S().copy()
            d = S().copy()
            ids = a[a["eligible"]]["student_id"]
            m = d["student_id"].isin(ids)
            tenth = m & (d["class"] == 10)
            d.loc[m & (d["class"] < 10), "class"] += 1
            d.loc[tenth, "status"] = "Passed out"
            st.session_state.students = d
            if tenth.sum():
                new_al = pd.DataFrame({"name": d.loc[tenth, "name"], "batch": dt.date.today().year, "now": "Class 10 pass-out",
                                       "city": "", "donation_inr": 0, "mentor": False})
                st.session_state.alumni = pd.concat([st.session_state.alumni, new_al], ignore_index=True)
            st.session_state.promo_history.append((dt.datetime.now(), int(m.sum())))
            st.success(f"Promoted {int(m.sum())} students.")
        if st.session_state.promo_backup is not None and st.button("Undo last promotion"):
            st.session_state.students = st.session_state.promo_backup
            st.session_state.promo_backup = None
            st.success("Reverted.")
        if st.session_state.promo_history:
            st.write("History:", st.session_state.promo_history)
    with t2:
        sid = student_picker("tc_sid")
        if sid:
            reason = st.selectbox("Reason", ["Parent relocation", "Change of school", "Admission elsewhere", "Other"])
            dest = st.text_input("Destination school")
            fee_due = st.session_state.fees.set_index("student_id").loc[sid]
            due = int(fee_due["annual_fee"] - fee_due["paid"]) if sid in st.session_state.fees["student_id"].values else 0
            st.write(f"Fee dues (demo data): INR {due:,}")
            if st.button("Issue TC and mark transferred", disabled=due > 0):
                n = len(st.session_state.transfers) + 1
                st.session_state.transfers.loc[n] = [f"TC/{dt.date.today().year}/{n:04d}", sid, dt.date.today(), reason, dest]
                st.session_state.students.loc[st.session_state.students["student_id"] == sid, "status"] = "Transferred"
                st.success("TC issued. Generate the PDF from Certificate Generator.")
            if due > 0:
                st.caption("TC is blocked until dues are cleared (common school policy; change if yours differs).")
        st.dataframe(st.session_state.transfers)


def page_disciplinary():
    st.header("Disciplinary Records")
    demo_banner("Sample incidents; the late-arrival and absence signals below are real.")
    ss = st.session_state
    with st.form("disc"):
        sid = student_picker("disc_sid")
        c = st.columns(3)
        cat = c[0].selectbox("Category", ["Late to class", "Uniform", "Bullying", "Phone misuse", "Fighting", "Disrespect", "Cheating", "Other"])
        sev = c[1].selectbox("Severity", ["Low", "Medium", "High"])
        act = c[2].selectbox("Action", ["Warning", "Parent call", "Counselling", "Detention", "Suspension"])
        det = st.text_area("Details")
        ok = st.form_submit_button("Log incident")
    if ok and sid:
        ss.disciplinary.loc[len(ss.disciplinary)] = [dt.date.today(), sid, cat, sev, det, act, st.session_state.get("role")]
        st.success("Logged.")
    d = ss.disciplinary
    a, b = st.columns(2)
    if len(d):
        a.plotly_chart(px.histogram(d, x="category", color="severity", title="Incidents by category"))
        rep = d["student_id"].value_counts().reset_index()
        rep = rep[rep["count"] >= 2]
        b.write("Repeat incidents (2+)")
        b.dataframe(rep)
    st.dataframe(d)
    st.subheader("Early signals from attendance data")
    sig = active()[(active()["late_arrivals_last_30d"] >= 5) | (active()["absences_last_30d"] >= 7)]
    st.dataframe(sig[["student_id", "name", "class", "section", "late_arrivals_last_30d", "absences_last_30d"]])
    st.caption("Treat these as prompts for a conversation, not as misconduct.")


# --------------------------------------------------------------------------------------
# PAGES: ACADEMIC
# --------------------------------------------------------------------------------------
def page_homework():
    st.header("Homework")
    real_banner("Submission rates are real; assigned homework list is created in-session.")
    ss = st.session_state
    t1, t2 = st.tabs(["Assign", "Submission analytics"])
    with t1:
        with st.form("hw"):
            c = st.columns(4)
            cl = c[0].selectbox("Class", [8, 9, 10])
            sec = c[1].selectbox("Section", ["A", "B", "C", "D", "All"])
            sub = c[2].selectbox("Subject", [SUBJ_LABEL[s] for s in SUBJECTS])
            mm = c[3].number_input("Max marks", 5, 100, 10)
            title = st.text_input("Title")
            due = st.date_input("Due date", dt.date.today() + dt.timedelta(days=3))
            ok = st.form_submit_button("Assign")
        if ok and title:
            ss.homework.loc[len(ss.homework)] = [cl, sec, sub, title, dt.date.today(), due, mm]
        st.dataframe(ss.homework)
    with t2:
        d = class_filter("hw")
        c = st.columns(3)
        c[0].metric("Avg submission %", f"{d['assignment_submission_pct'].mean():.1f}")
        c[1].metric("Avg days late", f"{d['avg_days_late'].mean():.2f}")
        c[2].metric("Below 60% submission", int((d["assignment_submission_pct"] < 60).sum()))
        st.plotly_chart(px.box(d, x="section", y="assignment_submission_pct", color="class", title="Submission by section"))
        st.plotly_chart(px.scatter(d, x="assignment_submission_pct", y="avg_halfyearly", trendline="ols" if _has_statsmodels() else None,
                                   title="Submission vs marks", opacity=.5))
        st.write("Students to nudge (low submission):")
        st.dataframe(d[d["assignment_submission_pct"] < 60][["student_id", "name", "class", "section", "assignment_submission_pct", "avg_days_late"]])


def _has_statsmodels():
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


def page_exams():
    st.header("Exams (online + offline)")
    ss = st.session_state
    t1, t2, t3 = st.tabs(["Results analytics", "Offline mark entry", "Online test"])
    with t1:
        real_banner()
        d = class_filter("ex")
        subj = st.selectbox("Subject", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s])
        tr = pd.DataFrame({"Exam": [EXAM_LABEL[e] for e in EXAMS], "Average": [d[f"{subj}_{e}"].mean() for e in EXAMS],
                           "Pass %": [(d[f"{subj}_{e}"] >= 33).mean() * 100 for e in EXAMS]})
        c = st.columns(2)
        c[0].plotly_chart(px.line(tr, x="Exam", y="Average", markers=True, title=f"{SUBJ_LABEL[subj]} trend"))
        c[1].plotly_chart(px.histogram(d, x=f"{subj}_halfyearly", nbins=20, title="Half-yearly score distribution"))
        top = d.nlargest(10, "avg_halfyearly")[["student_id", "name", "class", "section", "avg_halfyearly"]]
        st.write("Top 10")
        st.dataframe(top)
    with t2:
        d = class_filter("mk", with_all=False)
        subj = st.selectbox("Subject ", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s], key="mk_s")
        ex = st.selectbox("Exam", EXAMS, format_func=lambda e: EXAM_LABEL[e], key="mk_e")
        col = f"{subj}_{ex}"
        grid = d[["student_id", "name", col]].copy()
        ed = st.data_editor(grid, disabled=["student_id", "name"], key=f"grid_{col}_{len(d)}",
                            column_config={col: st.column_config.NumberColumn(min_value=0, max_value=100)})
        if st.button("Save marks"):
            S_ = S().set_index("student_id")
            S_.loc[ed["student_id"], col] = ed[col].values
            ss.students = S_.reset_index()[S().columns]
            st.success("Saved. Averages elsewhere will use these values (avg_halfyearly column is not auto-recomputed).")
    with t3:
        st.write("Build a quick MCQ test and take it.")
        with st.form("q", clear_on_submit=True):
            q = st.text_input("Question")
            opts = [st.text_input(f"Option {i + 1}") for i in range(4)]
            ans = st.selectbox("Correct option", [1, 2, 3, 4])
            if st.form_submit_button("Add question") and q and all(opts):
                ss.online_quiz.append({"q": q, "opts": opts, "ans": ans - 1})
        if ss.online_quiz:
            picks = []
            for i, it in enumerate(ss.online_quiz):
                picks.append(st.radio(f"{i + 1}. {it['q']}", it["opts"], index=None, key=f"quiz_{i}"))
            if st.button("Submit test"):
                sc = sum(1 for p, it in zip(picks, ss.online_quiz) if p == it["opts"][it["ans"]])
                st.success(f"Score: {sc}/{len(ss.online_quiz)}")
            if st.button("Clear test"):
                ss.online_quiz = []
                st.rerun()


def page_lesson_planner():
    st.header("Lesson Planner")
    st.caption("Template-based generator (no API key needed). Swap `generate_plan` for a Gemini/Claude call when you add an API key.")
    d = active()
    c = st.columns(4)
    cl = c[0].selectbox("Class", [8, 9, 10])
    subj = c[1].selectbox("Subject", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s])
    topic = c[2].text_input("Topic", "Introduction")
    mins = c[3].selectbox("Period length (min)", [30, 40, 45])
    cls = d[d["class"] == cl]
    weak = (cls[f"{subj}_halfyearly"] < 40).mean() * 100
    st.info(f"Real signal: {weak:.0f}% of Class {cl} scored below 40 in {SUBJ_LABEL[subj]} (half-yearly). "
            f"{'Include a remedial warm-up.' if weak > 15 else 'Standard pacing is fine.'}")
    if st.button("Generate plan"):
        a, b, c3, e = int(mins * .15), int(mins * .3), int(mins * .35), int(mins * .2)
        plan = f"""# {SUBJ_LABEL[subj]} · Class {cl} · {topic}
**Duration:** {mins} min

1. **Hook / recall ({a} min):** 2 quick questions from last lesson{'; add a remedial warm-up for weaker students' if weak > 15 else ''}.
2. **Teach ({b} min):** Explain {topic} using one real-life example and a board diagram.
3. **Guided practice ({c3} min):** Pairs solve 3 graded problems (easy, medium, challenge).
4. **Exit check ({e} min):** 3-question quiz; collect slips to find who needs follow-up.

**Homework:** 5 questions (2 easy, 2 medium, 1 challenge).
**Differentiation:** Peer buddy for students below 40; extension task for students above 80."""
        st.session_state.lesson_plans.append(plan)
    for p in reversed(st.session_state.lesson_plans):
        st.markdown(p)
        st.divider()
    if st.session_state.lesson_plans:
        st.download_button("Download plans (Markdown)", "\n\n".join(st.session_state.lesson_plans), "lesson_plans.md")


def page_assessments():
    st.header("Assessments")
    real_banner()
    d = class_filter("as")
    g = d["grade"].value_counts().reindex(["A1", "A2", "B1", "B2", "C1", "C2", "D", "E"]).fillna(0).reset_index()
    st.plotly_chart(px.bar(g, x="grade", y="count", title="Grade distribution (illustrative 8-band scale)"))
    fail = pd.DataFrame({SUBJ_LABEL[s]: [(d[f"{s}_halfyearly"] < 33).sum()] for s in SUBJECTS}).T.reset_index()
    fail.columns = ["Subject", "Below pass mark"]
    c = st.columns(2)
    c[0].plotly_chart(px.bar(fail, x="Subject", y="Below pass mark", title="Fails by subject (half-yearly)"))
    cm = d[[f"{s}_halfyearly" for s in SUBJECTS]].rename(columns=lambda x: SUBJ_LABEL[x.replace("_halfyearly", "")]).corr()
    c[1].plotly_chart(px.imshow(cm, text_auto=".2f", title="Subject correlation"))
    st.subheader("Improvement / decline")
    dd = d.assign(change=d["score_trend"])
    c = st.columns(2)
    c[0].write("Biggest improvers")
    c[0].dataframe(dd.nlargest(8, "change")[["student_id", "name", "class", "section", "change"]], hide_index=True)
    c[1].write("Biggest decliners")
    c[1].dataframe(dd.nsmallest(8, "change")[["student_id", "name", "class", "section", "change"]], hide_index=True)


def page_digital_eval():
    st.header("Digital Evaluation")
    st.caption("Evaluators enter question-wise marks; totals and remarks are stored with an audit trail. (Scan upload/annotation needs a PDF viewer component; add later.)")
    sid = student_picker("de_sid")
    if sid is None:
        return
    c = st.columns(3)
    subj = c[0].selectbox("Subject", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s])
    ex = c[1].selectbox("Exam", EXAMS, format_func=lambda e: EXAM_LABEL[e])
    nq = c[2].number_input("Questions", 3, 20, 6)
    scheme = pd.DataFrame({"question": [f"Q{i + 1}" for i in range(int(nq))], "max": 10, "awarded": 0.0})
    ed = st.data_editor(scheme, key=f"de_{sid}_{subj}_{ex}_{nq}", disabled=["question"])
    over = ed[ed["awarded"] > ed["max"]]
    total = float(ed["awarded"].sum())
    mx = float(ed["max"].sum())
    st.metric("Total", f"{total:.1f} / {mx:.0f}")
    rec = S().set_index("student_id").loc[sid, f"{subj}_{ex}"]
    if pd.notna(rec) and mx > 0:
        pct = total / mx * 100
        st.write(f"Recorded mark in system: **{rec}**, digital evaluation: **{pct:.0f}%**")
        if abs(pct - rec) > 10:
            st.warning("Mismatch above 10 points: send for re-evaluation.")
    rem = st.text_input("Remarks")
    if st.button("Save evaluation"):
        if len(over):
            st.error("Awarded marks exceed max for some questions.")
        else:
            st.session_state.digital_eval.loc[len(st.session_state.digital_eval)] = [sid, subj, ex, ed["awarded"].tolist(), total, st.session_state.get("role"), rem]
            st.success("Saved.")
    st.dataframe(st.session_state.digital_eval)


def page_library():
    st.header("Library")
    ss = st.session_state
    t1, t2 = st.tabs(["Catalogue & issue", "Reading vs performance (real)"])
    with t1:
        demo_banner("Catalogue is simulated.")
        bk = ss.library_books
        c = st.columns(3)
        c[0].metric("Titles", len(bk))
        c[1].metric("Copies", int(bk["copies"].sum()))
        c[2].metric("Issued", int(bk["issued"].sum()))
        with st.form("issue"):
            sid = student_picker("lib_sid")
            book = st.selectbox("Book", bk["book_id"] + " · " + bk["title"])
            ok = st.form_submit_button("Issue (14 days)")
        if ok and sid:
            bid = book.split(" · ")[0]
            row = bk[bk["book_id"] == bid].iloc[0]
            if row["issued"] >= row["copies"]:
                st.error("No copies available.")
            else:
                ss.library_books.loc[bk["book_id"] == bid, "issued"] += 1
                ss.library_issues.loc[len(ss.library_issues)] = [dt.date.today(), sid, bid, dt.date.today() + dt.timedelta(days=14), False]
                st.success("Issued.")
        st.dataframe(ss.library_issues)
        st.dataframe(ss.library_books)
    with t2:
        real_banner()
        d = active()
        d = d.assign(visits_band=pd.cut(d["library_visits_month"], [-1, 0, 2, 5, 20], labels=["0", "1-2", "3-5", "6+"]))
        g = d.groupby("visits_band", observed=True)["avg_halfyearly"].mean().reset_index()
        st.plotly_chart(px.bar(g, x="visits_band", y="avg_halfyearly", title="Marks by monthly library visits"))
        st.caption("Correlation only: motivated students may simply visit more.")


def page_study_center():
    st.header("Study Center (remedial batches)")
    real_banner("Groups are built from half-yearly marks, attendance and assignments.")
    d = class_filter("sc", with_all=True)
    c = st.columns(3)
    subj = c[0].selectbox("Subject", SUBJECTS, format_func=lambda s: SUBJ_LABEL[s])
    cut = c[1].slider("Weak if below", 20, 60, 40)
    size = c[2].number_input("Batch size", 5, 40, 15)
    weak = d[d[f"{subj}_halfyearly"] < cut].sort_values(f"{subj}_halfyearly")
    st.write(f"{len(weak)} students need support in {SUBJ_LABEL[subj]}")
    weak = weak.assign(batch=[f"Batch {i // int(size) + 1}" for i in range(len(weak))],
                       attendance_flag=np.where(weak["attendance_pct"] < 80, "Low attendance", ""),
                       internet=weak["home_internet"])
    st.dataframe(weak[["batch", "student_id", "name", "class", "section", f"{subj}_halfyearly", "attendance_flag", "internet"]])
    nonet = int((weak["internet"] == "No").sum())
    if nonet:
        st.info(f"{nonet} of these students have no home internet: give printed material, not online-only resources.")
    st.download_button("Download batch list", weak.to_csv(index=False), f"study_center_{subj}.csv")
    if len(weak):
        st.plotly_chart(px.bar(weak["batch"].value_counts().sort_index().reset_index(), x="batch", y="count", title="Batch sizes"))


def competency(score):
    if pd.isna(score):
        return None
    return "Beginner" if score < 40 else "Progressing" if score < 60 else "Proficient" if score < 80 else "Advanced"


def page_cbc():
    st.header("CBC / Competency-Based Academics")
    real_banner("Holistic profile mixes marks with aptitude, participation, assignments and activities.")
    sid = student_picker("cbc_sid")
    if sid is None:
        return
    r = S().set_index("student_id").loc[sid]
    dims = {
        "Subject knowledge": r["avg_halfyearly"],
        "Participation": (r["participation_score"] or 0) * 20,
        "Work habits": r["assignment_submission_pct"],
        "Reasoning": r["aptitude_logical"],
        "Creativity": r["aptitude_creative"],
        "Communication": r["aptitude_verbal"],
        "Co-curricular": min(100, r["num_activities"] * 33),
    }
    vals = [0 if pd.isna(v) else v for v in dims.values()]
    c = st.columns(2)
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=list(dims) + [list(dims)[0]], fill="toself"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), title=f"{r['name']}: competency radar")
    c[0].plotly_chart(fig)
    c[1].dataframe(pd.DataFrame({"Competency": list(dims), "Score": vals, "Level": [competency(v) for v in vals]}), hide_index=True)
    st.write(f"Activity: **{r['extracurricular']}**, counsellor sessions: **{int(r['counsellor_sessions'])}**")
    st.caption("Competency bands are illustrative. Map to your board's rubric (NEP/NCF) before report cards.")
    st.subheader("Class view: competency levels")
    d = class_filter("cbcc")
    lv = d["avg_halfyearly"].map(competency).value_counts().reindex(["Beginner", "Progressing", "Proficient", "Advanced"]).fillna(0).reset_index()
    st.plotly_chart(px.bar(lv, x="avg_halfyearly", y="count", title="Subject knowledge levels"))


def page_ptm():
    st.header("Parent-Teacher Meetings")
    real_banner("Attendance at past PTMs (out of 4) is real; slots booked here are session-only.")
    d = class_filter("ptm")
    c = st.columns(3)
    c[0].metric("Avg PTMs attended / 4", f"{d['ptm_attended_of_4'].mean():.2f}")
    c[1].metric("Never attended", int((d["ptm_attended_of_4"] == 0).sum()))
    c[2].metric("Attended all 4", int((d["ptm_attended_of_4"] == 4).sum()))
    g = d.groupby("ptm_attended_of_4")["avg_halfyearly"].mean().reset_index()
    st.plotly_chart(px.bar(g, x="ptm_attended_of_4", y="avg_halfyearly", title="Marks by PTM attendance"))
    st.subheader("Priority invites (low PTM attendance and at risk)")
    pr = d[(d["ptm_attended_of_4"] <= 1) & (d["at_risk"] == 1)]
    st.dataframe(pr[["student_id", "name", "class", "section", "ptm_attended_of_4", "avg_halfyearly"]])
    st.subheader("Book a slot")
    with st.form("ptmf"):
        sid = student_picker("ptm_sid", d)
        slot = st.selectbox("Slot", [f"{h}:00" for h in range(9, 16)] + [f"{h}:30" for h in range(9, 16)])
        day = st.date_input("Date", dt.date.today() + dt.timedelta(days=7))
        ag = st.text_input("Agenda", "Progress review")
        ok = st.form_submit_button("Book")
    if ok and sid:
        key = f"{day} {slot}"
        if key in set(st.session_state.ptm["slot"]):
            st.error("Slot already booked.")
        else:
            st.session_state.ptm.loc[len(st.session_state.ptm)] = [key, sid, "Class Teacher", ag, "Confirmed"]
    st.dataframe(st.session_state.ptm)


# --------------------------------------------------------------------------------------
# PAGES: FINANCE
# --------------------------------------------------------------------------------------
def page_fees():
    st.header("Fees & Accounts")
    demo_banner("Fee amounts/payments simulated; link a real payment gateway (Razorpay/PayTM) later.")
    ss = st.session_state
    f = ss.fees.merge(S()[["student_id", "name", "section", "status"]], on="student_id")
    f = f[f["status"] == "Active"]
    f["due"] = f["annual_fee"] - f["paid"]
    c = st.columns(4)
    c[0].metric("Billed", f"₹{f['annual_fee'].sum():,}")
    c[1].metric("Collected", f"₹{f['paid'].sum():,}")
    c[2].metric("Collection %", f"{f['paid'].sum() / f['annual_fee'].sum() * 100:.1f}%")
    c[3].metric("Outstanding", f"₹{f['due'].sum():,}")
    t1, t2, t3 = st.tabs(["Collect fee", "Defaulters + predictor", "Reports"])
    with t1:
        with st.form("fee"):
            sid = student_picker("fee_sid")
            amt = st.number_input("Amount (INR)", 100, 100000, 5000, 100)
            mode = st.selectbox("Mode", ["UPI", "Cash", "Card", "Cheque", "Net banking"])
            ok = st.form_submit_button("Record payment")
        if ok and sid:
            i = ss.fees.index[ss.fees["student_id"] == sid][0]
            ss.fees.at[i, "paid"] = min(ss.fees.at[i, "annual_fee"], ss.fees.at[i, "paid"] + amt)
            ss.fee_receipts.loc[len(ss.fee_receipts)] = [dt.datetime.now(), sid, amt, mode]
            st.success("Recorded.")
        st.dataframe(ss.fee_receipts)
    with t2:
        f["risk_score"] = (0.5 * (f["due"] / f["annual_fee"]) + 0.3 * (f["days_overdue"] / 90) + 0.2 * (f["instalments_missed"] / 3)).clip(0, 1)
        top = f[f["due"] > 0].sort_values("risk_score", ascending=False).head(50)
        st.dataframe(top[["student_id", "name", "class", "due", "days_overdue", "instalments_missed", "risk_score"]].style.format({"risk_score": "{:.0%}"}))
        st.caption("Rule-based score from payment behaviour only (not family background). Train a model once you have real payment history.")
        top["reminder"] = top.apply(lambda r: f"Dear Parent, fee of Rs {r['due']:,} for {r['name']} is pending. Kindly pay at the earliest. - {SCHOOL_NAME}", axis=1)
        st.download_button("Reminder messages (CSV)", top.to_csv(index=False), "fee_reminders.csv")
    with t3:
        g = f.groupby("class")[["annual_fee", "paid", "due"]].sum().reset_index()
        st.plotly_chart(px.bar(g.melt("class"), x="class", y="value", color="variable", barmode="group", title="Fees by class"))
        st.download_button("Fee ledger (Excel)", to_excel_bytes({"ledger": f}), "fee_ledger.xlsx")


def page_payroll():
    st.header("Payroll & HR")
    demo_banner("Staff list and salary rules are placeholders. Edit percentages to match your policy.")
    c = st.columns(4)
    hra = c[0].slider("HRA % of basic", 0, 40, 20)
    da = c[1].slider("DA % of basic", 0, 30, 10)
    pf = c[2].slider("PF % of basic (employee)", 0, 15, 12)
    pt = c[3].number_input("Professional tax", 0, 500, 200)
    st.session_state.staff = st.data_editor(st.session_state.staff, num_rows="dynamic", key="staff_ed")
    s = st.session_state.staff.copy()
    s["gross"] = (s["basic"] * (1 + hra / 100 + da / 100)).round(0)
    s["lop"] = (s["gross"] / 26 * s["leaves_taken"].clip(lower=0)).round(0)
    s["pf"] = (s["basic"] * pf / 100).round(0)
    s["net"] = s["gross"] - s["lop"] - s["pf"] - pt
    st.metric("Monthly payroll (net)", f"₹{int(s['net'].sum()):,}")
    st.dataframe(s)
    st.plotly_chart(px.bar(s.groupby("designation")["net"].sum().reset_index(), x="designation", y="net", title="Payroll by designation"))
    who = st.selectbox("Payslip for", s["emp_id"] + " · " + s["name"])
    r = s[s["emp_id"] == who.split(" · ")[0]].iloc[0]
    slip = (f"PAYSLIP {dt.date.today():%B %Y}\n{SCHOOL_NAME}\n{r['name']} ({r['emp_id']}, {r['designation']})\n"
            f"Gross: {r['gross']:.0f}\nLOP: -{r['lop']:.0f}\nPF: -{r['pf']:.0f}\nProf. tax: -{pt}\nNET PAY: {r['net']:.0f}\n")
    st.code(slip)
    st.download_button("Download payslip (txt)", slip, f"payslip_{r['emp_id']}.txt")
    st.download_button("Payroll register (Excel)", to_excel_bytes({"payroll": s}), "payroll.xlsx")


def page_inventory():
    st.header("Inventory")
    demo_banner()
    ss = st.session_state
    ss.inventory = st.data_editor(ss.inventory, num_rows="dynamic", key="inv_ed")
    inv = ss.inventory.copy()
    inv["value"] = inv["qty"] * inv["unit_cost"]
    low = inv[inv["qty"] <= inv["reorder_level"]]
    c = st.columns(2)
    c[0].metric("Stock value", f"₹{int(inv['value'].sum()):,}")
    c[1].metric("Items to reorder", len(low))
    if len(low):
        st.warning("Reorder now:")
        st.dataframe(low)
    st.plotly_chart(px.bar(inv, x="item", y="value", title="Stock value by item"))


def page_assets():
    st.header("Asset Management")
    demo_banner()
    a = st.session_state.assets.copy()
    age = (pd.Timestamp.today() - a["purchase_date"]).dt.days / 365.25
    a["age_years"] = age.round(1)
    a["book_value"] = (a["cost"] * (1 - (age / a["life_years"]).clip(0, 1))).round(0)
    a["status"] = np.where(age >= a["life_years"], "Fully depreciated", "In use")
    c = st.columns(3)
    c[0].metric("Cost", f"₹{int(a['cost'].sum()):,}")
    c[1].metric("Book value", f"₹{int(a['book_value'].sum()):,}")
    c[2].metric("Replace soon", int((age >= a["life_years"] * .85).sum()))
    st.dataframe(a)
    st.plotly_chart(px.bar(a, x="asset", y=["cost", "book_value"], barmode="group", title="Cost vs book value (straight-line)"))
    with st.expander("Add asset"):
        with st.form("addasset"):
            c = st.columns(3)
            nm = c[0].text_input("Asset")
            cat = c[1].text_input("Category")
            cost = c[2].number_input("Cost", 0, 10_000_000, 10000)
            life = st.number_input("Life (years)", 1, 30, 5)
            loc = st.text_input("Location")
            if st.form_submit_button("Add") and nm:
                st.session_state.assets.loc[len(st.session_state.assets)] = [nm, cat, pd.Timestamp.today(), cost, life, loc]
                st.rerun()


# --------------------------------------------------------------------------------------
# PAGES: OPERATIONS
# --------------------------------------------------------------------------------------
def build_routes():
    ss = st.session_state
    if "routes" in ss:
        return
    rng = np.random.default_rng(7)
    depot = ss.depot
    d = active()
    riders = d[d["commute_minutes"] > 8]["student_id"].tolist()
    routes = {}
    n_buses = 8
    for b in range(n_buses):
        ang = 2 * math.pi * b / n_buses
        stops = []
        for k in range(6, 0, -1):
            rad = 0.012 * k + rng.normal(0, .002)
            jitter = rng.normal(0, .12)
            stops.append((depot[0] + rad * math.sin(ang + jitter), depot[1] + rad * math.cos(ang + jitter)))
        routes[f"Bus {b + 1}"] = {"stops": stops + [depot], "students": []}
    for i, sid in enumerate(riders):
        routes[f"Bus {i % n_buses + 1}"]["students"].append(sid)
    ss.routes = routes


def nearest_neighbour(depot, stops):
    left = stops[:]
    path, cur = [], depot
    while left:
        nxt = min(left, key=lambda s: haversine(cur, s))
        path.append(nxt)
        left.remove(nxt)
        cur = nxt
    return path


def path_len(pts):
    return sum(haversine(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def page_transport():
    st.header("Transport & GPS")
    demo_banner("Routes and bus positions are simulated. Real tracking needs a GPS device/driver app posting to an API.")
    build_routes()
    ss = st.session_state
    t1, t2, t3 = st.tabs(["Live map", "Routes & riders", "Route optimizer"])
    with t1:
        auto = st.toggle("Auto-update (3 s)", value=False)

        def panel():
            if auto:
                ss.gps_tick += 1
            elif st.button("Advance one tick"):
                ss.gps_tick += 1
            rows = []
            for name, r in ss.routes.items():
                pts = r["stops"]
                t = (ss.gps_tick * 0.25) % (len(pts) - 1)
                i = int(t)
                f = t - i
                lat = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f
                lon = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f
                rows.append({"lat": lat, "lon": lon, "size": 140, "color": "#e63946", "label": name})
                for s in pts[:-1]:
                    rows.append({"lat": s[0], "lon": s[1], "size": 30, "color": "#457b9d", "label": "stop"})
            rows.append({"lat": ss.depot[0], "lon": ss.depot[1], "size": 220, "color": "#2a9d8f", "label": "School"})
            m = pd.DataFrame(rows)
            st.map(m, latitude="lat", longitude="lon", size="size", color="color")
            st.caption("Red = buses, blue = stops, green = school. Map centre is a placeholder; change `ss.depot` to your campus.")
        if auto and hasattr(st, "fragment"):
            st.fragment(run_every=3)(panel)()
        else:
            panel()
    with t2:
        tab = pd.DataFrame({"Bus": list(ss.routes), "Stops": [len(r["stops"]) - 1 for r in ss.routes.values()],
                            "Students": [len(r["students"]) for r in ss.routes.values()]})
        tab["Seat load (cap 45)"] = (tab["Students"] / 45 * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(tab)
        st.plotly_chart(px.bar(tab, x="Bus", y="Students", title="Riders per bus (assigned by simulation)"))
        sel = st.selectbox("Bus", list(ss.routes))
        sub = S()[S()["student_id"].isin(ss.routes[sel]["students"])]
        st.dataframe(sub[["student_id", "name", "class", "section", "commute_minutes"]])
        st.caption("Riders = students whose commute is over 8 min in your data; assignment to buses is simulated.")
    with t3:
        sel = st.selectbox("Route", list(ss.routes), key="opt_bus")
        stops = ss.routes[sel]["stops"][:-1]
        shuffled = stops[:]
        np.random.default_rng(1).shuffle(shuffled)
        cur = path_len(shuffled + [ss.depot])
        opt = path_len(nearest_neighbour(ss.depot, shuffled) + [ss.depot])
        c = st.columns(3)
        c[0].metric("Unplanned order", f"{cur:.1f} km")
        c[1].metric("Optimised (nearest neighbour)", f"{opt:.1f} km")
        c[2].metric("Saving", f"{(1 - opt / cur) * 100:.0f}%")
        st.caption("Greedy heuristic: fast and decent. Use OR-Tools for capacity/time-window constraints.")


def page_hostel():
    st.header("Hostel")
    demo_banner("Rooms are simulated. Students with the longest commute in your data are seeded as residents.")
    h = st.session_state.hostel
    occ = h["occupants"].map(len)
    c = st.columns(3)
    c[0].metric("Beds", int(h["capacity"].sum()))
    c[1].metric("Occupied", int(occ.sum()))
    c[2].metric("Vacant", int(h["capacity"].sum() - occ.sum()))
    view = h.assign(occupied=occ, occupants=h["occupants"].map(lambda x: ", ".join(x)))
    st.plotly_chart(px.bar(view, x="room", y="occupied", title="Occupancy by room"))
    st.dataframe(view)
    st.subheader("Allocate")
    with st.form("alloc"):
        sid = student_picker("host_sid")
        free = [r for r, o, cap in zip(h["room"], occ, h["capacity"]) if o < cap]
        room = st.selectbox("Room", free)
        ok = st.form_submit_button("Allocate")
    if ok and sid:
        if any(sid in o for o in h["occupants"]):
            st.error("Already allocated.")
        else:
            i = h.index[h["room"] == room][0]
            st.session_state.hostel.at[i, "occupants"] = h.at[i, "occupants"] + [sid]
            st.rerun()
    far = active().nlargest(10, "commute_minutes")[["student_id", "name", "class", "commute_minutes"]]
    st.write("Suggested candidates (longest commute, real data):")
    st.dataframe(far)


def page_health():
    st.header("Student Health Records")
    demo_banner("Sample records. Health data is sensitive: restrict to nurse/principal and log access (DPDP Act).")
    h = st.session_state.health.copy()
    h["bmi"] = (h["weight_kg"] / (h["height_cm"] / 100) ** 2).round(1)
    h["bmi_band"] = pd.cut(h["bmi"], [0, 14.5, 22, 27, 100], labels=["Low", "Normal", "High", "Very high"])
    c = st.columns(3)
    c[0].metric("Records", len(h))
    c[1].metric("Vaccination due/unknown", int((h["vaccination"] != "Up to date").sum()))
    c[2].metric("Vision below 6/9", int(h["vision"].isin(["6/12", "6/18"]).sum()))
    a, b = st.columns(2)
    a.plotly_chart(px.pie(h, names="vision", title="Vision screening"))
    b.plotly_chart(px.histogram(h, x="bmi_band", title="BMI bands (screening only, age-adjusted charts needed)"))
    st.dataframe(h.merge(S()[["student_id", "name", "class", "section"]], on="student_id"))
    with st.expander("Add / update record"):
        with st.form("hl"):
            sid = student_picker("hl_sid")
            c = st.columns(3)
            ht = c[0].number_input("Height (cm)", 90, 200, 150)
            wt = c[1].number_input("Weight (kg)", 15.0, 120.0, 42.0)
            vis = c[2].selectbox("Vision", ["6/6", "6/9", "6/12", "6/18"])
            vac = st.selectbox("Vaccination", ["Up to date", "Due", "Unknown"])
            if st.form_submit_button("Save") and sid:
                hh = st.session_state.health
                hh = hh[hh["student_id"] != sid]
                hh.loc[len(hh)] = [sid, ht, wt, vis, vac, dt.date.today()]
                st.session_state.health = hh
                st.rerun()


# --------------------------------------------------------------------------------------
# PAGES: ATTENDANCE
# --------------------------------------------------------------------------------------
def add_scan(code, method, device):
    d = S()
    code = code.strip()
    if code not in set(d["student_id"]):
        return False, "Unknown ID"
    log = st.session_state.scan_log
    today = dt.date.today()
    already = ((log["student_id"] == code) & (pd.to_datetime(log["time"]).dt.date == today)).any() if len(log) else False
    st.session_state.scan_log.loc[len(log)] = [dt.datetime.now(), code, method, device]
    return True, ("Already marked today (logged as repeat punch)" if already else "Marked present")


def scan_panel(method, device, help_text):
    st.caption(help_text)
    code = st.text_input("Scan / type ID and press Enter", key=f"scan_{method}")
    if code:
        ok, msg = add_scan(code, method, device)
        (st.success if ok else st.error)(f"{code}: {msg}")
    log = st.session_state.scan_log
    mine = log[log["method"] == method]
    c = st.columns(2)
    c[0].metric("Scans today", int(len(mine[pd.to_datetime(mine["time"]).dt.date == dt.date.today()])) if len(mine) else 0)
    c[1].metric("Unique students today", int(mine["student_id"].nunique()) if len(mine) else 0)
    st.dataframe(mine.tail(20))


def page_biometric():
    st.header("Biometric (ZKTeco)")
    st.caption("ZKTeco devices push punches to an ADMS endpoint or export an attendance log (.dat/.csv). "
               "Streamlit cannot receive device pushes directly: run a small FastAPI receiver that writes to your DB, "
               "or import the exported log here.")
    t1, t2 = st.tabs(["Import device log", "Simulate punches"])
    with t1:
        up = st.file_uploader("ZKTeco export (CSV/TXT: user_id, datetime, verify, status)", type=["csv", "txt", "dat"])
        if up:
            raw = pd.read_csv(up, sep=None, engine="python", header=None)
            raw = raw.iloc[:, :4]
            raw.columns = ["user_id", "datetime", "verify", "status"][:raw.shape[1]]
            raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
            raw["user_id"] = raw["user_id"].astype(str).str.strip()
            ok = raw[raw["user_id"].isin(set(S()["student_id"])) & raw["datetime"].notna()]
            st.write(f"{len(ok)} of {len(raw)} rows matched a student_id (device PIN must equal student_id).")
            st.dataframe(raw.head(20))
            if st.button("Add matched punches to log") and len(ok):
                add = pd.DataFrame({"time": ok["datetime"], "student_id": ok["user_id"], "method": "Biometric", "device": "ZKTeco"})
                st.session_state.scan_log = pd.concat([st.session_state.scan_log, add], ignore_index=True)
                st.success("Added.")
    with t2:
        n = st.slider("Simulate morning punches", 50, 800, 300, 50)
        if st.button("Run simulation"):
            rng = np.random.default_rng()
            d = active().sample(min(n, len(active())), random_state=None)
            base = dt.datetime.combine(dt.date.today(), dt.time(7, 30))
            add = pd.DataFrame({"time": [base + dt.timedelta(minutes=int(m)) for m in rng.integers(0, 60, len(d))],
                                "student_id": d["student_id"].values, "method": "Biometric", "device": "ZKTeco"})
            st.session_state.scan_log = pd.concat([st.session_state.scan_log, add], ignore_index=True)
            st.success(f"{len(d)} punches simulated.")
    mine = st.session_state.scan_log
    st.dataframe(mine[mine["method"] == "Biometric"].tail(20))


def page_qr():
    st.header("QR Code Attendance")
    t1, t2 = st.tabs(["Scan", "Generate ID cards"])
    with t1:
        scan_panel("QR", "Phone camera or USB QR scanner", "A USB/Bluetooth scanner types the code like a keyboard: click the box and scan.")
    with t2:
        if not HAS_QR:
            st.error('pip install "qrcode[pil]"')
            return
        d = class_filter("qrc", with_all=False)
        sid = student_picker("qr_sid", d)
        if sid:
            img = qrcode.make(sid)
            b = io.BytesIO()
            img.save(b, format="PNG")
            st.image(b.getvalue(), width=180, caption=sid)
        if st.button("Build ZIP of QR codes for this class/section"):
            zb = io.BytesIO()
            with zipfile.ZipFile(zb, "w") as z:
                for s in d["student_id"]:
                    bb = io.BytesIO()
                    qrcode.make(s).save(bb, format="PNG")
                    z.writestr(f"{s}.png", bb.getvalue())
            st.download_button("Download ZIP", zb.getvalue(), "qr_codes.zip")


def page_rfid():
    st.header("RFID / Barcode Attendance")
    scan_panel("RFID/Barcode", "Gate reader", "RFID readers and barcode scanners act like keyboards. Program each card/barcode with the student_id.")
    st.caption("Reuse the same reader at the bus door to mark boarding. Add a second log table keyed by bus.")


def page_att_analytics():
    st.header("Attendance Analytics")
    real_banner("Month-level attendance from your dataset, plus today's scans from this session.")
    d = class_filter("aa")
    c = st.columns(4)
    c[0].metric("Avg attendance %", f"{d['attendance_pct'].mean():.1f}")
    c[1].metric("Below 75%", int((d["attendance_pct"] < 75).sum()))
    c[2].metric("Dropped >5 pts vs last month", int(((d["attendance_pct"] - d["prev_month_attendance_pct"]) < -5).sum()))
    c[3].metric("Avg late arrivals / 30d", f"{d['late_arrivals_last_30d'].mean():.1f}")
    a, b = st.columns(2)
    g = d.groupby(["class", "section"])[["prev_month_attendance_pct", "attendance_pct"]].mean().reset_index()
    g["cs"] = g["class"].astype(str) + g["section"]
    a.plotly_chart(px.bar(g.melt(["class", "section", "cs"]), x="cs", y="value", color="variable", barmode="group", title="Previous vs current attendance"))
    b.plotly_chart(px.scatter(d, x="attendance_pct", y="avg_halfyearly", color="at_risk", title="Attendance vs marks", opacity=.5))
    st.subheader("Chronic absentees")
    ch = d[d["attendance_pct"] < 75].sort_values("attendance_pct")
    st.dataframe(ch[["student_id", "name", "class", "section", "attendance_pct", "absences_last_30d"]])
    log = st.session_state.scan_log
    if len(log):
        t = log[pd.to_datetime(log["time"]).dt.date == dt.date.today()]
        st.subheader("Today (from scans)")
        c = st.columns(3)
        c[0].metric("Present", int(t["student_id"].nunique()))
        c[1].metric("Absent", int(len(active()) - t["student_id"].nunique()))
        c[2].metric("Present %", f"{t['student_id'].nunique() / len(active()) * 100:.1f}")
        by = t.groupby("method")["student_id"].nunique().reset_index()
        st.plotly_chart(px.bar(by, x="method", y="student_id", title="Unique students by method"))


# --------------------------------------------------------------------------------------
# PAGES: PORTALS + FOUNDER
# --------------------------------------------------------------------------------------
def page_parent_view():
    st.header("Parent View")
    st.caption("What a parent app would show. In production, a parent can see only their own child (login-linked).")
    sid = student_picker("pv_sid")
    if sid is None:
        return
    r = S().set_index("student_id").loc[sid]
    c = st.columns(4)
    c[0].metric("Average %", f"{r['avg_halfyearly']:.1f}" if pd.notna(r["avg_halfyearly"]) else "-")
    c[1].metric("Attendance %", f"{r['attendance_pct']:.1f}" if pd.notna(r["attendance_pct"]) else "-")
    c[2].metric("Assignments %", f"{r['assignment_submission_pct']:.0f}" if pd.notna(r["assignment_submission_pct"]) else "-")
    f = st.session_state.fees
    due = int(f[f["student_id"] == sid].eval("annual_fee - paid").iloc[0]) if sid in set(f["student_id"]) else 0
    c[3].metric("Fee due (demo)", f"₹{due:,}")
    if pd.notna(r["avg_halfyearly"]):
        tr = pd.DataFrame([{"Subject": SUBJ_LABEL[s], "Exam": EXAM_LABEL[e], "Marks": r[f"{s}_{e}"]} for s in SUBJECTS for e in EXAMS])
        st.plotly_chart(px.line(tr, x="Exam", y="Marks", color="Subject", markers=True, title="Marks across exams"))
        al = build_alerts(S()[S()["student_id"] == sid])
        if len(al):
            st.warning("Things to discuss with the teacher: " + "; ".join(al["alert"]))
        else:
            st.success("No concerns flagged.")


def page_business():
    st.header("Business Plan: School SaaS")
    st.caption("Your plan from this chat: sell to Indian schools at about ₹100 per student per year.")
    t1, t2, t3 = st.tabs(["Unit economics", "Competitor benchmark", "Roadmap"])
    with t1:
        c = st.columns(3)
        schools = c[0].number_input("Schools signed", 1, 500, 5)
        students = c[1].number_input("Avg students per school", 100, 5000, 1500, 100)
        price = c[2].number_input("Price per student / year (INR)", 20, 500, 100, 10)
        c = st.columns(3)
        cac = c[0].number_input("Acquisition cost per school (INR)", 0, 500000, 40000, 5000)
        sup = c[1].number_input("Support cost per school / month (INR)", 0, 100000, 6000, 500)
        infra = c[2].number_input("Cloud + SMS/WhatsApp per student / year (INR)", 0, 100, 12)
        rev = schools * students * price
        cost = schools * cac + schools * sup * 12 + schools * students * infra
        c = st.columns(3)
        c[0].metric("Annual revenue", f"₹{rev:,.0f}")
        c[1].metric("Annual cost (year 1)", f"₹{cost:,.0f}")
        c[2].metric("Year-1 margin", f"₹{rev - cost:,.0f}")
        ratio = rev / max(cost, 1)
        st.progress(min(1.0, ratio / 2), text=f"Revenue / cost = {ratio:.2f}x")
        sweep = pd.DataFrame({"schools": range(1, 31)})
        sweep["margin"] = sweep["schools"] * students * price - (sweep["schools"] * cac + sweep["schools"] * sup * 12 + sweep["schools"] * students * infra)
        st.plotly_chart(px.line(sweep, x="schools", y="margin", title="Year-1 margin vs number of schools"))
        st.caption("Costs ignore your own salary, taxes and payment delays (schools often pay late). Model those before committing.")
    with t2:
        comp = pd.DataFrame({"Plan": ["Starter", "Standard", "Premium"], "Monthly (INR)": [1999, 2499, 3599], "Max students": [300, 500, 1000]})
        comp["Effective INR / student / year"] = (comp["Monthly (INR)"] * 12 / comp["Max students"]).round(1)
        st.dataframe(comp)
        st.write("From the Multi School ERP pricing page you shared. At ₹100 you are above these at full capacity, so lead with the AI early-warning layer, not a bundle of 50 modules.")
        feat = pd.DataFrame({
            "Area": ["Core ERP (fees, attendance, exams)", "Biometric / GPS / RFID", "AI risk + early warning", "Stream guidance", "Explainable alerts to parents"],
            "Big ERP": ["Yes", "Yes", "Claimed, no detail", "Rare", "Rare"],
            "Your wedge": ["Basic", "Phase 2", "Phase 1 focus", "Phase 1 focus", "Phase 1 focus"]})
        st.dataframe(feat)
    with t3:
        phases = {
            "Phase 0 (wk 1-3): Validate": ["Interview 10 principals/teachers", "Pick ONE painful problem", "Get 2-3 pilot schools"],
            "Phase 1 (wk 3-10): MVP": ["CSV import + student records", "Principal dashboard", "Role-based logins", "Risk flags + alerts"],
            "Phase 2 (mo 3-5): Pilot": ["Run free pilot with real data", "WhatsApp/SMS reminders", "Measure outcomes (flags that proved right)"],
            "Phase 3 (parallel): Trust": ["DPDP consent + privacy policy", "Encrypted backups + audit log", "Data processing agreement"],
            "Phase 4 (mo 5-9): First customers": ["Convert pilots to paid", "Demo deck + 5-min video", "Support routine (WhatsApp helpdesk, SLA)"],
            "Phase 5 (mo 9-18): Scale": ["Fees/report-card modules", "Integrations with existing ERPs", "Hire support, then sales"],
        }
        done = total = 0
        for ph, items in phases.items():
            with st.expander(ph, expanded=False):
                for it in items:
                    k = f"rm_{ph}_{it}"
                    v = st.checkbox(it, key=k)
                    total += 1
                    done += int(v)
        st.progress(done / total if total else 0, text=f"{done}/{total} tasks done")
    st.caption("Do not copy competitors' claims ('2000+ schools', 'bank-grade'). Only publish numbers you can prove.")


# --------------------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------------------
PAGE_FUNCS = {
    "Principal Dashboard": page_principal_dashboard, "AI Risk & Alerts": page_ai_risk, "Stream Guidance": page_stream,
    "Admissions & Enrollment": page_admissions, "Roles & Permissions": page_roles, "Import / Export": page_import_export,
    "Certificate Generator": page_certificates, "Front Office Desk": page_frontdesk, "Alumni": page_alumni,
    "Promotion & Transfer": page_promotion, "Disciplinary Records": page_disciplinary,
    "Homework": page_homework, "Exams": page_exams, "Lesson Planner": page_lesson_planner, "Assessments": page_assessments,
    "Digital Evaluation": page_digital_eval, "Library": page_library, "Study Center": page_study_center,
    "CBC Academics": page_cbc, "Parent-Teacher Meetings": page_ptm,
    "Fees & Accounts": page_fees, "Payroll & HR": page_payroll, "Inventory": page_inventory, "Asset Management": page_assets,
    "Transport & GPS": page_transport, "Hostel": page_hostel, "Student Health Records": page_health,
    "Biometric (ZKTeco)": page_biometric, "QR Code": page_qr, "RFID / Barcode": page_rfid, "Attendance Analytics": page_att_analytics,
    "Parent View": page_parent_view, "Business Plan": page_business,
}


def main():
    st.sidebar.title(f"🏫 {SCHOOL_NAME}")
    load_students()
    init_state()
    role = st.sidebar.selectbox("Sign in as (demo)", ROLES, key="role")
    perm = st.session_state.perm
    allowed = set(perm.index[perm[role]]) if role in perm.columns else set()
    if role == "Super Admin":
        allowed = set(ALL_PAGES)
    groups = {g: [p for p in ps if p in allowed] for g, ps in GROUPS.items()}
    groups = {g: ps for g, ps in groups.items() if ps}
    if not groups:
        st.error("This role has no pages enabled.")
        return
    group = st.sidebar.selectbox("Module", list(groups), key="group")
    page = st.sidebar.radio("Page", groups[group], key=f"page_{group}")
    a = active()
    st.sidebar.divider()
    st.sidebar.caption(f"{len(a):,} active students · {a['avg_halfyearly'].notna().sum():,} with marks")
    st.sidebar.caption("Student data is personal data of minors: collect only what you need, get parental consent (DPDP Act 2023).")
    PAGE_FUNCS[page]()


main()
