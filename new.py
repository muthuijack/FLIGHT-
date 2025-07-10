import streamlit as st
import joblib 
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import os

# --- Load Models ---
rf_model = joblib.load("C:\\Users\\HP\\Downloads\\rf_model.pkl")
gbr_model = joblib.load("C:\\Users\\HP\\Downloads\\gbr_model.pkl")

models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gbr_model
}

# --- Page Setup ---
st.set_page_config(page_title="Lift Coefficient Predictor", layout="wide")
st.title("✈️ Lift Coefficient (Cl) Predictor")

# --- Initialize session state defaults ---
if "model_used" not in st.session_state:
    st.session_state.model_used = list(models.keys())[0]
if "cl_pred" not in st.session_state:
    st.session_state.cl_pred = 0.0
if "airspeed" not in st.session_state:
    st.session_state.airspeed = 0.0

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Predict Cl", "📊 Visualize 3D Plot", "📐 Measurement & Report"])

# --- Tab 1: Prediction ---
with tab1:
    st.header("🔧 Enter Airfoil Parameters")

    camber = st.number_input("Camber (0.00 - 0.09)", 0.0, 0.1, 0.03, step=0.01, format="%.4f", key="camber")
    thickness = st.number_input("Thickness (0.06 - 0.15)", 0.0, 0.2, 0.1, step=0.01, format="%.4f", key="thickness")
    aoa = st.number_input("Angle of Attack (°)", -10.0, 30.0, 5.0, step=0.5, format="%.2f", key="aoa")
    reynolds = st.number_input("Reynolds Number", 1e5, 5e6, 2e6, step=1e5, format="%.0f", key="re")

    selected_model_name = st.selectbox("Select model for prediction:", list(models.keys()), key="model")
    selected_model = models[selected_model_name]

    if st.button("🚀 Predict Cl"):
        input_data = np.array([[camber, thickness, aoa, reynolds]])
        cl_pred = selected_model.predict(input_data)[0]
        st.success(f"🔮 Predicted Cl using {selected_model_name}: {cl_pred:.3f}")
        if cl_pred > 1.5:
            st.warning("⚠️ High Cl — possible near-stall condition.")

        st.session_state.cl_pred = cl_pred
        st.session_state.model_used = selected_model_name

        st.subheader("✈️ Optional: Estimate Required Speed")
        rho = st.number_input("Air Density (kg/m³)", min_value=0.5, max_value=1.5, value=1.225, step=0.01)
        S = st.number_input("Wing Area (m²)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
        L = st.number_input("Lift Force Required (Newtons)", min_value=500.0, max_value=50000.0, value=9800.0, step=100.0)

        try:
            V = (2 * L / (rho * S * cl_pred)) ** 0.5
            st.session_state.airspeed = V
            st.info(f"💨 Estimated Required Airspeed: **{V:.2f} m/s**")

            if V < 20:
                st.warning("⚠️ Very low speed — may result in insufficient lift. Consider increasing wing area or airspeed.")
            elif V > 100:
                st.warning("⚠️ Very high speed — structural integrity may be at risk. Consider reducing AoA or camber.")

        except ZeroDivisionError:
            st.error("Cannot compute speed — Cl must be greater than 0.")

# --- Tab 2: 3D Plot ---
with tab2:
    st.header("📊 3D Surface Visualization")
    if st.button("🌐 Generate 3D Plot"):
        aoa_range = np.linspace(-5, 20, 25)
        re_range = np.linspace(1e5, 5e6, 25)
        aoa_grid, re_grid = np.meshgrid(aoa_range, re_range)
        cl_grid = np.zeros_like(aoa_grid)

        for i in range(aoa_grid.shape[0]):
            for j in range(aoa_grid.shape[1]):
                input_row = np.array([[st.session_state.camber, st.session_state.thickness,
                                       aoa_grid[i, j], re_grid[i, j]]])
                cl_grid[i, j] = models[st.session_state.model_used].predict(input_row)[0]

        fig = go.Figure(data=[go.Surface(z=cl_grid, x=aoa_grid, y=re_grid, colorscale='Viridis')])
        fig.update_layout(
            title="3D Surface: Cl vs AoA vs Reynolds",
            scene=dict(
                xaxis_title="Angle of Attack (°)",
                yaxis_title="Reynolds Number",
                zaxis_title="Predicted Cl"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.session_state.last_inputs = {
            "camber": st.session_state.camber,
            "thickness": st.session_state.thickness,
            "aoa": st.session_state.aoa,
            "re": st.session_state.re
        }

        try:
            fig.write_image("3d_plot.png")
            st.session_state.plot_image_path = "3d_plot.png"
        except Exception as e:
            st.warning("⚠️ Kaleido is not installed. Cannot save plot as PNG. Run: pip install -U kaleido")

# --- Tab 3: Measurement & Report ---
with tab3:
    st.header("📐 Auto Measurements & PDF Report")

    if "last_inputs" in st.session_state:
        inputs = st.session_state.last_inputs
        st.markdown(f"**Camber**: {inputs['camber']}")
        st.markdown(f"**Thickness**: {inputs['thickness']}")
        st.markdown(f"**AoA**: {inputs['aoa']}°")
        st.markdown(f"**Reynolds Number**: {inputs['re']:.0f}")
        st.markdown(f"**Predicted Cl**: {st.session_state.cl_pred:.3f}")
        st.markdown(f"**Estimated Speed**: {st.session_state.airspeed:.2f} m/s")

        uploaded_image = st.file_uploader("📷 Optionally upload an airfoil reference image", type=["png", "jpg", "jpeg"])

        if st.button("📝 Export PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Lift Coefficient Prediction Report", ln=True, align="C")
            pdf.ln(10)

            for k, v in inputs.items():
                pdf.cell(200, 10, txt=f"{k.title()}: {v}", ln=True)

            pdf.cell(200, 10, txt=f"Predicted Cl: {st.session_state.cl_pred:.3f}", ln=True)
            pdf.cell(200, 10, txt=f"Estimated Speed: {st.session_state.airspeed:.2f} m/s", ln=True)
            pdf.cell(200, 10, txt=f"Model Used: {st.session_state.model_used}", ln=True)

            if st.session_state.airspeed < 20:
                pdf.cell(200, 10, txt="⚠️ Warning: Speed too low. Consider increasing wing area or camber.", ln=True)
            elif st.session_state.airspeed > 100:
                pdf.cell(200, 10, txt="⚠️ Warning: Speed too high. Reduce AoA or camber for safety.", ln=True)

            if "plot_image_path" in st.session_state and os.path.exists(st.session_state.plot_image_path):
                pdf.image(st.session_state.plot_image_path, x=10, y=None, w=180)

            if uploaded_image:
                img_path = f"uploaded_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(img_path, "wb") as f:
                    f.write(uploaded_image.read())
                pdf.image(img_path, x=10, y=None, w=180)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"cl_report_{timestamp}.pdf"
            pdf.output(report_name)

            with open(report_name, "rb") as f:
                st.download_button("📥 Download PDF", f, file_name=report_name, mime="application/pdf")
    else:
        st.warning("⚠️ Please first generate a 3D plot in Tab 2.")

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ by Jack Muthui · July 2025 · Powered by Streamlit & scikit-learn")
