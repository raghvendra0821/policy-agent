import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Policy Data Agent", layout="wide")
st.title("AI Policy Data Agent")
st.caption("NITI Aayog — Scheme Monitoring Dashboard")

@st.cache_data(ttl=0)
def fetch_data():
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {
        "api-key": "579b464db66ec23bdd0000015916930689ba433e6c0c2bc9660139da",
        "format": "json",
        "limit": "100"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        records = data.get("records", [])
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        st.error(f"API fetch failed: {e}")
        return pd.DataFrame()

def generate_dummy_data():
    import numpy as np
    np.random.seed(42)
    states = ["Uttar Pradesh","Bihar","Rajasthan","Madhya Pradesh",
              "Maharashtra","West Bengal","Gujarat","Odisha","Assam","Jharkhand"]
    data = {
        "state": states,
        "beneficiaries_registered": np.random.randint(50000, 500000, len(states)),
        "disbursement_amount_cr": np.round(np.random.uniform(100, 2000, len(states)), 2),
        "pending_cases": np.random.randint(100, 15000, len(states)),
        "scheme": ["PM-KISAN"] * len(states)
    }
    df = pd.DataFrame(data)
    return df

def detect_anomalies(df, col):
    mean = df[col].mean()
    std = df[col].std()
    df["anomaly"] = df[col].apply(
        lambda x: "High" if x > mean + 1.5*std
        else ("Low" if x < mean - 1.5*std else "Normal")
    )
    return df

def generate_report(df):
    top_pending = df.nlargest(3, "pending_cases")[["state","pending_cases"]].to_string(index=False)
    low_disbursement = df.nsmallest(3, "disbursement_amount_cr")[["state","disbursement_amount_cr"]].to_string(index=False)
    anomalies = df[df["anomaly"] != "Normal"][["state","anomaly"]].to_string(index=False)

    prompt = f"""
You are a senior policy analyst at NITI Aayog preparing a government scheme monitoring brief.

Based on the following data from PM-KISAN scheme monitoring:

Top 3 states with highest pending cases:
{top_pending}

Bottom 3 states by disbursement amount (Crore INR):
{low_disbursement}

Anomaly flags:
{anomalies}

Write a concise 150-word policy brief that:
1. Summarizes the key findings
2. Flags states needing urgent intervention
3. Gives 2 actionable recommendations

Use formal government report language.
"""
    response = model.generate_content(prompt)
    return response.text

# ── Sidebar ──
st.sidebar.header("Controls")
use_live = st.sidebar.checkbox("Use live data.gov.in API", value=False)
metric = st.sidebar.selectbox("Analyse metric", ["pending_cases","disbursement_amount_cr","beneficiaries_registered"])

# ── Load Data ──
st.subheader("Scheme Data")
if use_live:
    with st.spinner("Fetching from data.gov.in..."):
        df = fetch_data()
    if df.empty:
        st.warning("Live data unavailable — loading demo data.")
        df = generate_dummy_data()
else:
    df = generate_dummy_data()

df = detect_anomalies(df, metric)
st.dataframe(df, use_container_width=True)

# ── Anomaly Summary ──
col1, col2, col3 = st.columns(3)
col1.metric("Total States", len(df))
col2.metric("Anomalies Flagged", len(df[df["anomaly"] != "Normal"]))
col3.metric("Avg Pending Cases", int(df["pending_cases"].mean()))

# ── Chart ──
st.subheader(f"District Analysis — {metric}")
color_map = {"Normal": "#1D9E75", "High": "#E24B4A", "Low": "#EF9F27"}
fig = px.bar(
    df.sort_values(metric, ascending=False),
    x="state", y=metric,
    color="anomaly",
    color_discrete_map=color_map,
    title=f"{metric} by State"
)
st.plotly_chart(fig, use_container_width=True)

# ── AI Report ──
st.subheader("AI-Generated Policy Brief")
if st.button("Generate Policy Brief with Gemini"):
    with st.spinner("Gemini is writing the brief..."):
        report = generate_report(df)
    st.success("Report generated!")
    st.markdown(report)

st.caption("Built with LangGraph-style agent architecture | NITI Aayog Internship Demo")