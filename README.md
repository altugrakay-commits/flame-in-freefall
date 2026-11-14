# 🔥 Flame in Freefall: Microgravity Combustion Dashboard

## 📖 Overview
This project is a submission for the **NASA Space Apps Challenge 2026: Flame in Freefall**. It provides a physics-informed, AI-driven dashboard that summarizes, compares, and interprets microgravity combustion experiments to support fire safety in space exploration.

## 🚀 Key Features
1. **Experiment Explorer:** Visualizes raw time-series data (pixel intensity) extracted from NASA raw images, correlated with actual flame photographs from the ISS.
2. **AI Safety Insights (PINN):** Implements a **Physics-Informed Neural Network (PINN)** to predict flame intensity and extinction events, strictly obeying physical laws (bounded energy and non-negativity).
3. **Comparative Analysis:** Compares two distinct combustion regimes from the NASA PSI repository:
   - **SLICE (PSI-106):** Non-sooting, spherical methane flames (extinction limits).
   - **SPICE (PSI-107):** Highly sooting, elongated ethylene flames (smoke points).

## 🧠 The AI Architecture (PINN)
Standard neural networks suffer from "spectral bias" and often hallucinate unphysical values when modeling sparse scientific data. To solve this, this project utilizes a custom **Physics-Informed Neural Network (PINN)** built in PyTorch.

Key innovations in the model:
- **Fourier Feature Mapping:** Overcomes spectral bias, allowing the network to learn the sharp, high-frequency extinction events in the flame data.
- **Bounded Energy Constraints:** The custom loss function penalizes the network for predicting negative light or exceeding maximum physical brightness, ensuring the surrogate model remains physically grounded.

## 📊 Data Sources
- **SLICE (PSI-106):** NASA Physical Sciences Informatics (PSI) system. Test run GMT 079 (100% Methane).
- **SPICE (PSI-107):** NASA PSI system. Test run GMT 045 (Ethylene).

## 💻 Technical Stack
- **Language:** Python
- **Dashboard:** Streamlit, Plotly
- **Machine Learning:** PyTorch (Custom PINN architecture)
- **Data Processing:** Pandas, NumPy, OpenPyXL

## 👨‍🔬 Author
**Altug Remzi Akay, Ph.D.**
*Computational Simulation & Systems Optimization Engineer*