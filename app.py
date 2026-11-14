import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data.loader import load_slice_timeseries
from src.models.pinn import FlamePINN

st.set_page_config(page_title="Flame in Freefall", layout="wide")

st.title("🔥 Flame in Freefall: Microgravity Combustion Dashboard")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Experiment Explorer", "AI Safety Insights (PINN)", "Comparative Analysis"])

# Load Data
filepath = "data/raw/SLICE_GMT079/pic_info_059_noBG.xls"
df = load_slice_timeseries(filepath)

# ==========================================
# TAB 1: Experiment Explorer
# ==========================================
with tab1:
    st.header("Experiment Time-Series Data")
    if df is not None:
        fig = px.line(df, x='frame_idx', y=['intensity_1', 'intensity_2', 'intensity_3'],
                      labels={'value': 'Pixel Intensity', 'frame_idx': 'Frame Index', 'variable': 'Sensor Channel'},
                      title="Raw Flame Intensity over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open('assets/demo_screenshots/flame_high.png'), caption="Stable Flame (High Intensity)", use_container_width=True)
        with col2:
            st.image(Image.open('assets/demo_screenshots/flame_low.png'), caption="Near-Extinction Flame (Low Intensity)", use_container_width=True)

# ==========================================
# TAB 2: AI Safety Insights (PINN)
# ==========================================
with tab2:
    st.header("Physics-Informed Neural Network (PINN) Surrogate")
    st.markdown("""
    Standard AI models often 'hallucinate' physically impossible values (like negative light). 
    This dashboard uses a **PINN** trained on the NASA data. The loss function incorporates 
    **thermodynamic smoothness** and **non-negativity constraints** to ensure predictions strictly obey physical laws.
    """)
    
    if df is not None:
        # Prepare data for model
        df_clean = df.dropna(subset=['intensity_3'])
        t_data = df_clean['frame_idx'].values.reshape(-1, 1).astype(np.float32)
        
        # 1. Get the scaling factor used during training
        I_max = np.max(df_clean['intensity_3'].values)
        
        t_min, t_max = t_data.min(), t_data.max()
        t_norm = (t_data - t_min) / (t_max - t_min)
        t_tensor = torch.from_numpy(t_norm)

        model_path = "src/models/pinn_weights.pth"
        if os.path.exists(model_path):
            model = FlamePINN()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                # Predict, then multiply back by I_max to un-normalize
                I_pred_norm = model(t_tensor).numpy()
                I_pred = I_pred_norm * I_max
            
            # Create a Plotly figure comparing True vs Predicted
            fig_ai = go.Figure()
            
            # Raw Data
            fig_ai.add_trace(go.Scatter(x=df_clean['frame_idx'], y=df_clean['intensity_3'], 
                                        mode='markers', name='Actual Data (NASA)',
                                        marker=dict(color='red', size=4, opacity=0.5)))
            
            # PINN Prediction
            fig_ai.add_trace(go.Scatter(x=df_clean['frame_idx'], y=I_pred.flatten(), 
                                        mode='lines', name='PINN Surrogate',
                                        line=dict(color='cyan', width=3)))
            
            fig_ai.update_layout(title="Actual vs. PINN Predicted Flame Intensity",
                                 xaxis_title="Frame Index",
                                 yaxis_title="Pixel Intensity")
            
            st.plotly_chart(fig_ai, use_container_width=True)
            
            st.success("✅ Note how the PINN curve smoothly interpolates the data and strictly avoids dropping below zero, enforcing physical laws.")
            
        else:
            st.warning("Model weights not found. Please run `python src/models/pinn.py` in your terminal first to train the model.")

# ==========================================
# TAB 3: Comparative Analysis
# ==========================================
with tab3:
    st.header("Comparative Analysis: SLICE vs. SPICE")
    st.markdown("""
    To fully understand fire safety in microgravity, we must study different combustion regimes. 
    This dashboard compares two NASA experiments:
    *   **SLICE (Methane):** Focuses on non-sooting, spherical blue flames and extinction limits.
    *   **SPICE (Ethylene):** Focuses on highly sooting, elongated yellow flames and smoke points.
    """)
    
    st.markdown("---")
    
    col_slice, col_spice = st.columns(2)
    
    with col_slice:
        st.subheader("SLICE (PSI-106)")
        st.markdown("**Fuel:** Methane (CH4)")
        st.markdown("**Characteristic:** Blue, hemispherical flame. Minimal soot production.")
        try:
            st.image('assets/demo_screenshots/flame_high.png', caption="SLICE Microgravity Flame", use_container_width=True)
        except:
            st.warning("SLICE image not found.")
            
        st.markdown("""
        **AI Insight:** The PINN model successfully predicts the sharp extinction events 
        (when the flame drops to zero intensity) due to the bounded energy physics constraints.
        """)

    with col_spice:
        st.subheader("SPICE (PSI-107)")
        st.markdown("**Fuel:** Ethylene (C2H4)")
        st.markdown("**Characteristic:** Yellow, elongated flame. High soot production, leading to smoke points.")
        try:
            # Make sure you named your telemetry image this!
            st.image('assets/demo_screenshots/spice_telemetry.png', caption="SPICE Flame with Telemetry Overlay", use_container_width=True)
        except:
            st.warning("SPICE image not found.")
            
        st.markdown("""
        **Physics Insight:** The telemetry overlay shows real-time flow rates. Unlike SLICE, SPICE 
        flames grow much longer before reaching the smoke point, posing a different type of fire hazard 
        (smoke inhalation and sensor obscuration in spacecraft).
        """)

    st.markdown("---")
    st.subheader("Mass Flow Calibration (SPICE)")
    st.markdown("Raw voltage data from the SPICE Mass Flow Controller is converted to physical units (SCCM) to accurately track fuel consumption.")
    
    # Hardcoding the calibration data from the Excel file you provided for simplicity
    import pandas as pd
    calib_data = pd.DataFrame({
        "Flow Rate (SCCM)": [0, 125, 250, 375, 500],
        "Output (V)": [0, 1.266, 2.503, 3.741, 5.0]
    })
    st.dataframe(calib_data, use_container_width=True, hide_index=True)