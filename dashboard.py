# ---------- dashboard.py ----------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import re
from datetime import date

from intelligent_fitting_pipeline import run_pipeline
from monte_carlo_simulation import run_simulation_ui

st.set_page_config(layout="wide")
st.title("🧠 Probabilistic Valuation Model for Life Settlements")

# ------------------ Curve Fitting Section ------------------

st.header("📈 Life Table Curve Fitting (Gompertz & Weibull)")

def run_pipeline_uncached(image_bytes):
    with open("temp_life_table_image.png", "wb") as f:
        f.write(image_bytes)
    return run_pipeline("temp_life_table_image.png")

uploaded_file = st.file_uploader("Upload a life table image (PNG, JPG):", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    if "last_image_hash" not in st.session_state or image_hash != st.session_state.last_image_hash:
        try:
            fitted = run_pipeline_uncached(image_bytes)
            gom = fitted.get('gompertz')
            wei = fitted.get('weibull')
            fig = fitted.get('fig')

            if fig is None:
                st.error("❌ Failed to generate visualization chart. Please re-upload a clearer image.")
                st.stop()

            st.session_state["fit_fig"] = fig
            st.session_state.fitted_result = fitted
            st.session_state.last_image_hash = image_hash
            st.session_state["gom_a"] = gom[0]
            st.session_state["gom_b"] = gom[1]  # annual scale already per pipeline
            st.session_state["wei_shape"] = wei[0]
            st.session_state["wei_scale"] = wei[1]

            st.success("✅ Curve fitting successful!")
            st.markdown(f"**Gompertz (a, b):** `{gom[0]:.5f}, {gom[1]:.5f}` ")
            st.markdown(f"**Weibull (shape, scale):** `{wei[0]:.5f}, {wei[1]:.2f}` ")
            st.pyplot(fig)

        except Exception as e:
            st.session_state.fitted_result = None
            st.error("❌ GPT could not extract a valid life table. Please try a clearer image.")
            st.exception(e)

    elif st.session_state.fitted_result:
        gom = st.session_state.fitted_result['gompertz']
        wei = st.session_state.fitted_result['weibull']
        st.markdown(f"**Gompertz (a, b):** `{gom[0]:.5f}, {gom[1]:.5f}` ")
        st.markdown(f"**Weibull (shape, scale):** `{wei[0]:.5f}, {wei[1]:.2f}` ")
        st.pyplot(st.session_state["fit_fig"])

# ------------------ Helpers for Premium Streams ------------------

def _parse_amount_token(token: str) -> float:
    """'$1,035.19' -> 1035.19; '$-' / '-' / blank -> 0.0"""
    if token is None:
        return 0.0
    s = str(token).strip()
    if s == "" or s in {"-", "—", "–", "$-", "$—", "$–"}:
        return 0.0
    cleaned = re.sub(r"[^0-9.\-]", "", s)
    return float(cleaned) if cleaned not in {"", "-", "."} else 0.0

def _parse_pasted_premiums(text: str):
    """
    One amount per line, e.g.:
      $-
      $611.41
      $1,035.19
    Month 1 = first day of current month. Returns {month_index -> amount}.
    """
    if not text or not text.strip():
        return None
    rows = [ln for ln in text.splitlines() if ln.strip() != ""]
    amounts = [_parse_amount_token(ln) for ln in rows]
    return {i + 1: amt for i, amt in enumerate(amounts)}

def _parse_premium_upload(upload):
    """
    Accepts CSV/XLSX/TXT in one of these schemas:
      - month_index, amount
      - date, amount
      - single column of amounts (strings like '$-', '$1,035.19', or numerics)
    Missing/blank/NaN => 0. Returns {month_index:int -> amount:float}.
    """
    if upload is None:
        return None

    name = upload.name.lower()
    if name.endswith(".txt"):
        return _parse_pasted_premiums(upload.read().decode("utf-8"))

    if name.endswith(".csv"):
        df = pd.read_csv(upload)
    else:
        df = pd.read_excel(upload)  # needs openpyxl

    df.columns = [str(c).strip().lower() for c in df.columns]

    # month_index, amount
    if "month_index" in df.columns and "amount" in df.columns:
        df["month_index"] = pd.to_numeric(df["month_index"], errors="coerce")
        df["amount"] = df["amount"].apply(_parse_amount_token)
        df = df.dropna(subset=["month_index"])
        df = df[df["month_index"] >= 1]
        return {int(m): float(a) for m, a in df[["month_index", "amount"]].itertuples(index=False, name=None)}

    # date, amount
    date_cols = [c for c in df.columns if "date" in c]
    amt_cols = [c for c in df.columns if c in ("amount", "premium") or "amount" in c or "premium" in c]
    if date_cols and amt_cols:
        c_date, c_amt = date_cols[0], amt_cols[0]
        df[c_amt] = df[c_amt].apply(_parse_amount_token)
        df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
        df = df.dropna(subset=[c_date])
        today0 = date.today().replace(day=1)
        df["month_index"] = (df[c_date].dt.year - today0.year) * 12 + (df[c_date].dt.month - today0.month) + 1
        df = df[df["month_index"] >= 1]
        return {int(m): float(a) for m, a in df[["month_index", c_amt]].itertuples(index=False, name=None)}

    # single column of amounts
    if df.shape[1] == 1:
        col = df.columns[0]
        amounts = df[col].apply(_parse_amount_token).tolist()
        return {i + 1: float(v) for i, v in enumerate(amounts)}

    st.error("Premium file must be one of: (month_index,amount), (date,amount), a single column of amounts, or a .txt list (one per line).")
    return None

# ------------------ Monte Carlo Section ------------------

st.markdown("---")
st.header("📊 Policy Valuation with Monte Carlo Simulations")

if st.session_state.get("fitted_result"):
    col1, col2 = st.columns(2)

    with col1:
        face_value = st.number_input("Face Value ($)", value=1_000_000.00)
        cash_value = st.number_input("Cash Surrender Value ($)", value=0.00)
        premium_annual = st.number_input("Average Annual Premium (fallback) ($)", value=30_000.00)
        le_at_report = st.number_input("LE at Report Date (months)", value=72)
        le_report_date = st.date_input("LE Report Date")

    with col2:
        gom_a = st.number_input("Gompertz: Shape (a)", value=st.session_state.get("gom_a", 0.08))
        gom_b = st.number_input("Gompertz: Scale (b, annual)", value=st.session_state.get("gom_b", 0.01))
        wei_shape = st.number_input("Weibull: Shape", value=st.session_state.get("wei_shape", 5.0))
        wei_scale = st.number_input("Weibull: Scale (annual)", value=st.session_state.get("wei_scale", 10.0))

    st.markdown("### Premium Stream (optional)")
    premium_file = st.file_uploader(
        "Upload premium stream (CSV/XLSX/TXT). You may also paste below.",
        type=["csv", "xlsx", "txt"]
    )
    premium_paste = st.text_area("Paste one amount per line (use $- for blanks)", height=220,
                                 placeholder="$-\n$-\n$611.41\n$611.30\n...")

    premium_schedule = None
    if premium_paste.strip():
        premium_schedule = _parse_pasted_premiums(premium_paste)
    elif premium_file is not None:
        premium_schedule = _parse_premium_upload(premium_file)

    # Optional quick preview
    if premium_schedule:
        first = sorted(premium_schedule.items())[:12]
        last = sorted(premium_schedule.items())[-12:]
        st.caption("Parsed premium stream (first & last 12 months shown):")
        st.table(pd.DataFrame(first, columns=["Month", "Amount"]))
        st.table(pd.DataFrame(last, columns=["Month", "Amount"]))

    if st.button("▶️ Run Simulation"):
        gom_stats, wei_stats, fig, le_adjusted = run_simulation_ui(
            face_value,
            cash_value,
            premium_annual,     # fallback if no premium stream uploaded/pasted
            gom_a,
            gom_b,
            wei_shape,
            wei_scale,
            le_at_report,
            le_report_date.strftime("%Y-%m-%d"),
            premium_schedule=premium_schedule  # NEW: dict or None
        )
        st.pyplot(fig)

        st.subheader("💰 Net Present Policy Value as of Today")
        col_g, col_w = st.columns(2)
        col_g.metric("Gompertz Valuation", f"${gom_stats['interpolated']:,.0f}")
        col_w.metric("Weibull Valuation", f"${wei_stats['interpolated']:,.0f}")
        st.caption(f"📆 Adjusted LE as of today: **{le_adjusted:.1f} months**")
else:
    st.info("Please upload and fit a life table image first to populate Monte Carlo inputs.")
