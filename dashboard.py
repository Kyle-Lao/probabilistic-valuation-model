# ---------- dashboard.py ----------

import streamlit as st
from intelligent_fitting_pipeline import run_pipeline
from monte_carlo_simulation import run_simulation_ui
import matplotlib.pyplot as plt
import hashlib

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
            st.session_state["gom_b"] = gom[1]
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

# ------------------ Monte Carlo Section ------------------

st.markdown("---")
st.header("📊 Policy Valuation with Monte Carlo Simulations")

if st.session_state.get("fitted_result"):
    col1, col2 = st.columns(2)

    with col1:
        face_value = st.number_input("Face Value ($)", value=1000000)
        cash_value = st.number_input("Cash Value ($)", value=50000)
        premium_annual = st.number_input("Annual Premium ($)", value=30000)
        le_at_report = st.number_input("LE at Report Date (months)", value=72)
        le_report_date = st.date_input("LE Report Date")

    with col2:
        gom_a = st.number_input("Gompertz: Shape (a)", value=st.session_state.get("gom_a", 0.08))
        gom_b = st.number_input("Gompertz: Scale (b)", value=st.session_state.get("gom_b", 0.01))
        wei_shape = st.number_input("Weibull: Shape", value=st.session_state.get("wei_shape", 5.0))
        wei_scale = st.number_input("Weibull: Scale", value=st.session_state.get("wei_scale", 10.0))

    if st.button("▶️ Run Simulation"):
        gom_stats, wei_stats, fig, le_adjusted = run_simulation_ui(
            face_value,
            cash_value,
            premium_annual,
            gom_a,
            gom_b,
            wei_shape,
            wei_scale,
            le_at_report,
            le_report_date.strftime("%Y-%m-%d")
        )
        st.pyplot(fig)

        st.subheader("💰 Net Present Policy Value at Updated LE (Today)")
        col_g, col_w = st.columns(2)
        col_g.metric("Estimated Current Valuation based on Fitted Gompertz Curve", f"${gom_stats['interpolated']:,.0f}")
        col_w.metric("Estimated Current Valuation based on Fitted Weibull Curve", f"${wei_stats['interpolated']:,.0f}")
        st.caption(f"📆 Adjusted LE as of today: **{le_adjusted:.1f} months**")
else:
    st.info("Please upload and fit a life table image first to populate Monte Carlo inputs.")

