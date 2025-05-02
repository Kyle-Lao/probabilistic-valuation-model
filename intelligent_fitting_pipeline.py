import openai
import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dotenv import load_dotenv
from io import StringIO
import os

# --------------------------
# CONFIGURATION
# --------------------------
load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]

# --------------------------
# MODEL FUNCTIONS
# --------------------------
def gompertz_cdf(x, a, b):
    return 1 - np.exp(-(b / a) * (np.exp(a * x) - 1))

def weibull_cdf(x, shape, scale):
    return 1 - np.exp(- (x / scale) ** shape)

# --------------------------
# UTILITIES
# --------------------------
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def extract_csv_with_gpt(image_path):
    base64_img = encode_image(image_path)
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in actuarial modeling. Extract a clean CSV from this life table image. "
                    "The CSV must contain exactly two columns: Year and Survival, where Survival is a decimal from 0 to 1. "
                    "Return plain CSV text only — no quotes, no markdown formatting, no code blocks."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract the life table below as a CSV with columns Year and Survival."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]
            }
        ]
    )

    csv_raw = response.choices[0].message.content.strip("`\n ")
    df = pd.read_csv(StringIO(csv_raw))

    if df.shape[1] == 1 and 'csv' in df.columns[0].lower():
        print("⚠️ GPT returned CSV as single string column — re-parsing...")
        df = pd.read_csv(StringIO(df.columns[0]))

    return df

# --------------------------
# PREPROCESSING
# --------------------------
def preprocess_and_truncate(df):
    print("📊 GPT extracted columns:", list(df.columns))

    possible_surv = [col for col in df.columns if any(key in col.lower() for key in ["survival", "prob", "remain", "perc"])]
    if not possible_surv:
        print("❌ Could not find a survival column. Detected columns:", list(df.columns))
        raise ValueError("No survival-like column found in GPT-extracted table.")
    surv_col = possible_surv[0]

    possible_time = [col for col in df.columns if any(key in col.lower() for key in ["year", "age", "time"])]
    if not possible_time:
        print("❌ Could not find a time column. Detected columns:", list(df.columns))
        raise ValueError("No time-like column found in GPT-extracted table.")
    time_col = possible_time[0]

    df = df.rename(columns={surv_col: "Survival", time_col: "Year"})
    df = df.dropna()
    df["Survival"] = pd.to_numeric(df["Survival"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna()
    df = df[df["Survival"] <= 1.0]

    cutoff_idx = df[df["Survival"] < 0.01].index.min()
    if pd.notna(cutoff_idx):
        df = df.loc[:cutoff_idx]

    df["Mortality"] = 1 - df["Survival"]
    return df.reset_index(drop=True)

# --------------------------
# FITTING + PLOTTING
# --------------------------
def fit_models(x, y):
    gom_params, _ = curve_fit(gompertz_cdf, x, y, p0=[0.1, 0.1], maxfev=10000)
    wei_params, _ = curve_fit(weibull_cdf, x, y, p0=[1.5, 10.0], maxfev=10000)
    return gom_params, wei_params

def plot_fit(x, y, gom_params, wei_params, title="Mortality Fit"):
    x_fine = np.linspace(min(x), max(x), 300)
    gom_y = gompertz_cdf(x_fine, *gom_params)
    wei_y = weibull_cdf(x_fine, *wei_params)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='black', label='Observed Mortality')
    
    x_fine = np.linspace(min(x), max(x), 300)
    gom_y = gompertz_cdf(x_fine, *gom_params)
    wei_y = weibull_cdf(x_fine, *wei_params)
    
    ax.plot(x_fine, gom_y, '--', label='Gompertz Fit', color='blue')
    ax.plot(x_fine, wei_y, ':', label='Weibull Fit', color='red')
    
    ax.set_title(title)
    ax.set_xlabel("Life Expectancy (Years)")
    ax.set_ylabel("Cumulative Mortality")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    
    return fig


# --------------------------
# MAIN PIPELINE
# --------------------------
def run_pipeline(image_path):
    print("📤 Extracting data from image using GPT-4o...")
    df_raw = extract_csv_with_gpt(image_path)

    print("🧹 Preprocessing and truncating at <1% survival...")
    df = preprocess_and_truncate(df_raw)

    print("📈 Fitting Gompertz and Weibull curves...")
    x = df["Year"].values
    y = df["Mortality"].values
    gom, wei = fit_models(x, y)

    # Transform for display and simulation purposes
    gom_annual = [gom[0], gom[1] * 12]   # Scale adjusted to annual
    wei_annual = [wei[0], wei[1]]       # Scale stays annual

    print("\n✅ Final Parameters:")
    print(f"Gompertz: a = {gom_annual[0]:.5f}, b = {gom_annual[1]:.5f} (annual scale)")
    print(f"Weibull:  shape = {wei_annual[0]:.5f}, scale = {wei_annual[1]:.5f} (annual scale)")

    fig = plot_fit(x, y, gom, wei, title="Fitted Mortality Curves (Truncated at 1% Survival)")

    return {
        "gompertz": gom_annual,
        "weibull": wei_annual,
        "fig":fig
    }
