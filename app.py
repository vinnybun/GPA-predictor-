
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import re

st.set_page_config(page_title="GPA AI Dashboard", layout="wide")

st.title("🎓 GPA AI Dashboard")
st.markdown("Analyze • Simulate • Predict • Plan your academic performance")

GRADE_MAP = {"A":5,"B":4,"C":3,"D":2,"E":1,"F":0}

# =========================
# HELPER FUNCTIONS
# =========================
def map_semester(sem):
    sem = int(sem)
    year = (sem + 1) // 2
    sem_type = "Harmattan" if sem % 2 == 1 else "Rain"
    return f"Year {year} - {sem_type} Semester"

def extract_sem_num(label):
    nums = list(map(int, re.findall(r'\d+', label)))
    year = nums[0]
    sem = 1 if "Harmattan" in label else 2
    return (year * 2) - (2 - sem)

def calculate_gpa(df):
    return df.groupby("semester").apply(
        lambda x: (x["unit"] * x["grade_point"]).sum() / x["unit"].sum()
    ).reset_index(name="GPA")

def calculate_cgpa(df):
    total_units = df["unit"].sum()
    total_points = (df["unit"] * df["grade_point"]).sum()
    return round(total_points / total_units, 2)

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("⚙️ Input Settings")

mode = st.sidebar.radio("Input Method", ["Upload CSV", "Manual Entry"])

data = pd.DataFrame()

# =========================
# CSV INPUT
# =========================
if mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df["grade"] = df["grade"].str.upper()
        df["grade_point"] = df["grade"].map(GRADE_MAP)
        df["semester"] = df["semester"].apply(map_semester)
        data = df
        st.success("✅ CSV Loaded Successfully")

# =========================
# MANUAL INPUT
# =========================
else:
    st.sidebar.subheader("✍️ Manual Entry")
    num_courses = st.sidebar.number_input("Number of Courses", 1, 15, 3)

    rows = []
    for i in range(num_courses):
        course = st.sidebar.text_input(f"Course {i+1}", key=f"c{i}")
        unit = st.sidebar.number_input(f"Unit {i+1}", 1, 6, key=f"u{i}")
        grade = st.sidebar.selectbox(f"Grade {i+1}", ["A","B","C","D","E","F"], key=f"g{i}")

        year = st.sidebar.number_input(f"Year {i+1}", 1, 10, key=f"y{i}")
        sem_type = st.sidebar.selectbox(f"Semester Type {i+1}", ["Harmattan","Rain"], key=f"st{i}")

        semester = f"Year {year} - {sem_type} Semester"

        rows.append({
            "course": course,
            "unit": unit,
            "grade": grade,
            "grade_point": GRADE_MAP[grade],
            "semester": semester
        })

    data = pd.DataFrame(rows)

# =========================
# MAIN APP
# =========================
if not data.empty:

    gpa_df = calculate_gpa(data)
    cgpa = calculate_cgpa(data)

    gpa_df["sem_num"] = gpa_df["semester"].apply(extract_sem_num)
    gpa_df = gpa_df.sort_values("sem_num")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "📉 Trends",
        "🧪 What-if",
        "🤖 Prediction",
        "🎯 Target CGPA"
    ])

    # =========================
    # DASHBOARD
    # =========================
    with tab1:
        col1, col2 = st.columns(2)
        col1.metric("🎯 Current CGPA", cgpa)
        col2.metric("📚 Total Courses", len(data))

        st.dataframe(data)

        # Performance insight
        if len(gpa_df) > 1:
            if gpa_df["GPA"].iloc[-1] > gpa_df["GPA"].iloc[-2]:
                st.success("📈 Your performance is improving")
            else:
                st.warning("📉 Your performance is declining")

    # =========================
    # TRENDS
    # =========================
    with tab2:
        plt.figure()
        plt.plot(gpa_df["semester"], gpa_df["GPA"], marker='o')
        plt.xticks(rotation=45)
        plt.title("GPA Trend")
        plt.grid()
        st.pyplot(plt)

    # =========================
    # WHAT-IF
    # =========================
    with tab3:
        num = st.number_input("Future courses", 1, 10, 2)

        hypo = []
        for i in range(num):
            col1, col2, col3 = st.columns(3)
            with col1:
                course = st.text_input(f"Course {i}", key=f"h{i}")
            with col2:
                unit = st.number_input(f"Unit {i}", 1, 6, key=f"hu{i}")
            with col3:
                grade = st.selectbox(f"Grade {i}", ["A","B","C","D","E","F"], key=f"hg{i}")

            hypo.append({
                "course": course,
                "unit": unit,
                "grade": grade,
                "grade_point": GRADE_MAP[grade]
            })

        if st.button("Run Scenario"):
            new_df = pd.concat([data, pd.DataFrame(hypo)])
            new_cgpa = calculate_cgpa(new_df)

            st.metric("New CGPA", new_cgpa)

    # =========================
    # AI PREDICTION
    # =========================
    with tab4:
        X = gpa_df[["sem_num"]]
        y = gpa_df["GPA"]

        model = LinearRegression().fit(X,y)
        pred = model.predict([[X["sem_num"].max()+1]])[0]

        st.metric("Predicted Next GPA", round(pred,2))

    # =========================
    # TARGET CGPA
    # =========================
    with tab5:
        target = st.number_input("Target CGPA", 0.0, 5.0, step=0.1)

        total_units = data["unit"].sum()
        total_points = (data["unit"] * data["grade_point"]).sum()

        future_units = st.number_input("Next semester total units", 1, 30, 15)

        required_points = target * (total_units + future_units) - total_points
        required_gpa = required_points / future_units

        st.metric("Required GPA Next Semester", round(required_gpa,2))

        if required_gpa > 5:
            st.error("⚠️ Target not achievable")
        else:
            st.success("🎯 You can achieve this!")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    report = data.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download Report", report, "gpa_report.csv")

else:
    st.info("👈 Upload or input data to begin")

