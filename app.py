import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Market Predictor",
    layout="wide",
    page_icon="📊"
)

# =========================
# CUSTOM STYLING (THIS IS THE MAGIC)
# =========================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #FFFFFF;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1 style='text-align: center;'>📊 Market Predictor</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; color: gray;'>Predicting S&P 500 returns using macroeconomic indicators</p>",
    unsafe_allow_html=True
)

st.divider()

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 2])

# =========================
# INPUT PANEL (LEFT)
# =========================
with col1:
    st.markdown("### 📥 Input Variables")

    with st.container():
        inflation = st.slider("Inflation", 0.0, 300.0, 200.0)
        unemployment = st.slider("Unemployment Rate", 0.0, 15.0, 5.0)
        interest = st.slider("Interest Rate", 0.0, 10.0, 2.0)

    run = st.button("🚀 Run Model", use_container_width=True)

# =========================
# RESULTS PANEL (RIGHT)
# =========================
with col2:
    if run:
        # =========================
        # PREDICTION
        # =========================
        predicted_return = (
            0.0069 * inflation +
            0.0123 * unemployment +
            0.0242 * interest
        )

        st.markdown("### 📊 Prediction")

        st.metric(
            label="Expected Monthly Return",
            value=f"{predicted_return:.4f}",
            delta=None
        )

        st.divider()

        # =========================
        # INSIGHTS CARD
        # =========================
        st.markdown("### 🧠 Key Insights")

        st.markdown("""
<div class="card">
• Lagged unemployment is the most important feature<br>
• Linear Regression outperformed XGBoost<br>
• Macroeconomic variables have limited predictive power
</div>
""", unsafe_allow_html=True)

        st.divider()

        # =========================
        # VISUALS (CLEANER)
        # =========================
        st.markdown("### 📈 Model Visualizations")

        colA, colB = st.columns(2)

        with colA:
            st.image("images/model_comparison.png", use_container_width=True)

        with colB:
            st.image("images/feature_importance.png", use_container_width=True)

    else:
        st.markdown("""
        <div style='text-align: center; margin-top: 100px; color: gray;'>
        Click "Run Model" to generate predictions and insights
        </div>
        """, unsafe_allow_html=True)