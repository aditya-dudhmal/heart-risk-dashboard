"""Heart disease risk explorer.

Streamlit dashboard for a clinic team to explore which patient factors
track with heart disease in the UCI Cleveland dataset. For pattern
exploration and discussion — not a diagnostic tool.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Heart Disease Risk Explorer", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data/heart.csv", na_values="?").dropna()
    df["target"] = (df["num"] > 0).astype(int)
    df["sex"] = df["sex"].map({1: "Male", 0: "Female"})
    df["age_group"] = pd.cut(df["age"], bins=[0, 45, 55, 65, 120],
                             labels=["<45", "45-54", "55-64", "65+"])
    return df


df = load_data()

st.title("Heart Disease Risk Explorer")
st.write("Explore which patient factors track with heart disease across "
         f"{len(df)} clinic patients (UCI Cleveland dataset).")

# --- Sidebar filters ---
st.sidebar.header("Filters")
sex_choice = st.sidebar.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])
age_range = st.sidebar.slider("Age range", int(df["age"].min()), int(df["age"].max()),
                              (int(df["age"].min()), int(df["age"].max())))
filtered = df[df["sex"].isin(sex_choice) & df["age"].between(*age_range)]
st.sidebar.write(f"Showing {len(filtered)} patients")

# --- Headline metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", len(filtered))
c2.metric("Heart disease prevalence", f"{filtered['target'].mean():.0%}")
c3.metric("Average age", f"{filtered['age'].mean():.0f}")
c4.metric("Avg max heart rate", f"{filtered['thalach'].mean():.0f}")

# --- Charts ---
left, right = st.columns(2)

with left:
    st.subheader("Prevalence by age group")
    rate = filtered.groupby("age_group", observed=True)["target"].mean()
    fig, ax = plt.subplots()
    rate.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Share with heart disease")
    ax.set_xlabel("")
    st.pyplot(fig)

with right:
    st.subheader("Prevalence by sex")
    rate = filtered.groupby("sex")["target"].mean()
    fig, ax = plt.subplots()
    rate.plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_ylabel("Share with heart disease")
    ax.set_xlabel("")
    st.pyplot(fig)

st.subheader("Resting blood pressure vs cholesterol")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered, x="trestbps", y="chol", hue="target", ax=ax, alpha=0.7)
ax.set_xlabel("Resting blood pressure (mm Hg)")
ax.set_ylabel("Cholesterol (mg/dl)")
st.pyplot(fig)

# --- What the model leans on ---
st.subheader("What a model leans on")
st.write("A random forest trained on all patients ranks the measurements "
         "that best separate patients with and without heart disease. "
         "Chest pain type, number of major vessels, and thalassemia result "
         "typically carry the most signal in this dataset.")
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
X = pd.get_dummies(df[features], columns=["sex"])
model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, df["target"])
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
importance.plot(kind="barh", ax=ax)
ax.set_title("Random forest feature importance")
st.pyplot(fig)

st.caption("Data: UCI Machine Learning Repository, Cleveland heart disease dataset (303 patients).")
