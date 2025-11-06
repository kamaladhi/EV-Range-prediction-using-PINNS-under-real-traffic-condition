# ⚡ EV Range Prediction using Physics-Informed Neural Network (PINN)

### 🚗 Intelligent, Physics-Consistent Estimation of SOC, Power, and Range for Electric Vehicles

This project implements a **Physics-Informed Neural Network (PINN)** for **Electric Vehicle (EV) range prediction**.  
Unlike conventional machine learning models that rely purely on data, this architecture embeds **real vehicle physics** directly into the neural network, ensuring predictions are **physically consistent**, **interpretable**, and **reliable under unseen conditions**.

---

## 📘 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Physics Layer Details](#physics-layer-details)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Training Pipeline](#training-pipeline)
- [Outputs & Visualizations](#outputs--visualizations)
- [Real-World Applications](#real-world-applications)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🧠 Overview

Modern EVs often mispredict range due to dynamic real-world conditions — terrain, temperature, driver behavior, and energy losses.  
This project addresses that by combining **data-driven learning** with **physics-based modeling**, using a deep **LSTM + Attention + Physics-Layer** hybrid network.

The system predicts:
- 🔋 **State of Charge (SOC)** – remaining battery percentage  
- ⚙️ **Power Consumption (kW)** – instantaneous power draw  
- 🛣️ **Driving Range (km)** – estimated remaining range  

and enforces **realistic physical behavior** using embedded battery & vehicle dynamics equations.

---

## 🔑 Key Features

✅ **Physics-Informed Loss Integration**  
Learns battery power consumption and SOC under real physical constraints.

✅ **Hybrid Deep Learning Architecture**  
Combines **LSTM** (for sequential patterns), **Attention** (for interpretability), and **Dense Estimators** (for final range).

✅ **Realistic EV Physics**  
Models drag, rolling resistance, gravity, temperature effects, and drivetrain efficiency.

✅ **Visualization Suite**  
Auto-generates plots for model accuracy, physics validation, range trends, and attention heatmaps.

✅ **Explainable Predictions**  
Attention heatmaps reveal which timesteps most influenced range estimation.

✅ **Modular & Scalable**  
Easily adaptable for different EV datasets or route conditions.

---

## 🧩 Architecture

```text
Input: [velocity, acceleration, elevation, ambient temperature, ...] → (time-series sequence)

            ┌───────────────────────────────┐
            │   Input Projection Layer       │
            └─────────────┬─────────────────┘
                          │
                   ┌──────▼──────┐
                   │   LSTM      │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ Attention    │
                   └──────┬──────┘
          ┌────────────────────────────────┐
          │                                │
   ┌──────▼──────┐                 ┌───────▼──────┐
   │ SOC Head     │                 │ Power Head   │
   └──────┬───────┘                 └──────┬──────┘
          │                                │
          └──────────────┬─────────────────┘
                         ▼
                 Range Estimator + Physics Layer
                         │
                   Final Outputs:
         SOC (0–1), Power (kW), Range (km)
```

---

## ⚙️ Physics Layer Details

The **`PhysicsLayer`** enforces realistic constraints using vehicle parameters:

| Parameter | Description | Symbol |
|------------|--------------|--------|
| `battery_capacity` | Battery energy (kWh) | C |
| `vehicle_mass` | Vehicle weight (kg) | m |
| `drag_coeff` | Aerodynamic drag | Cd |
| `frontal_area` | Vehicle frontal area (m²) | A |
| `rolling_resistance` | Tire rolling resistance | Cr |
| `air_density` | Ambient air density | ρ |
| `gravity` | Gravitational acceleration | g |
| `efficiency` | Drivetrain efficiency | η |

**Power Equation:**
\[
P = \frac{(F_{drag} + F_{roll} + F_{grade} + F_{accel}) \cdot v}{\eta} + P_{aux}
\]

This power is compared to the neural prediction using **Huber loss**, ensuring the model’s outputs stay within physical realism.

---

## 🧮 Dataset & Preprocessing

Input data includes:
- Velocity (km/h)  
- Acceleration (m/s²)  
- Elevation grade (%)  
- Ambient temperature (°C)  
- Energy / power consumption (kW)  
- SOC over time

Data is normalized, cleaned, and serialized into:
```
processed_ev_data.pkl
```

Splits:
- 80% training
- 20% testing

Each sample is a **time-series window** of driving data.

---

## 🔧 Training Pipeline

**Training Command:**
```bash
python ev_pinn_train.py
```

**Workflow:**
1. Load dataset → create PyTorch `DataLoader`
2. Initialize EVRangePINN model
3. Compute combined data + physics losses
4. Apply learning rate scheduling & early stopping
5. Save best model weights and history

**Core Loss:**
\[
L_{total} = L_{data} + \lambda_{physics}L_{physics}
\]

where \( L_{physics} \) enforces:
- Power consistency  
- SOC bounds  
- Energy magnitude limits

---

## 📊 Outputs & Visualizations

All plots are auto-saved in `ev_pinn_plots/`.

| Plot | Description |
|------|--------------|
| **training_validation_physics_loss.png** | Convergence of training, validation, and physics loss |
| **prediction_accuracy_errors_combined.png** | SOC & Power accuracy + error histograms |
| **physics_analysis.png** | Model vs Physics power, vs velocity, elevation, temperature |
| **energy_consumption_heatmap.png** | Heatmap of energy consumption (kWh/km) vs speed & acceleration |
| **range_prediction_analysis.png** | Range distribution and correlations (Range–SOC, Range–Power) |
| **attention_heatmap.png** | Attention weights across timesteps for interpretability |

---

## 🌍 Real-World Applications

| Sector | Application | Impact |
|---------|--------------|--------|
| 🚘 **EV Manufacturers** | Real-time range prediction | Reduces driver range anxiety |
| 🔋 **Battery Management Systems** | SOC estimation improvement | Increases battery life accuracy |
| 🚛 **Fleet Operators** | Route energy optimization | Reduces operational cost |
| 🧪 **Research & Simulation** | Digital twin validation | Tests physics-ML hybrids |
| ⚡ **Charging Networks** | Demand forecasting | Smarter grid load management |

---

## 🧰 Installation

```bash
git clone https://github.com/<your-username>/EV-Range-Prediction-PINN.git
cd EV-Range-Prediction-PINN
pip install -r requirements.txt
```

**requirements.txt**
```
torch
numpy
matplotlib
seaborn
scikit-learn
pickle-mixin
```

---

## ▶️ Usage

1. Place your processed dataset in:
   ```
   data/processed_ev_data.pkl
   ```

2. Train the model:
   ```bash
   python ev_pinn_train.py
   ```

3. Generate visualizations:
   ```bash
   python ev_pinn_analysis.py
   ```

4. Results saved automatically in:
   ```
   ev_pinn_plots/
   ```

---

## 📂 Repository Structure



---

## 🚀 Future Improvements

- Incorporate **battery degradation physics** for aging-aware prediction  
- Add **multi-modal fusion** (GPS, weather, and driver data)  
- Extend to **real-time deployment on edge hardware** (NVIDIA Jetson / Raspberry Pi)  
- Integrate **reinforcement learning** for adaptive eco-driving suggestions  

---

## 📜 License

This project is licensed under the **MIT License** — free for research, development, and educational use.

---

### 👨‍💻 Author

**Jeeva Kamal**  

📧 [jeevakamal2005@gmail.com](mailto:jeevakamal2005@gmail.com)  
🌐 GitHub: [github.com/jeevakamal](https://github.com/jeevakamal)

---

### 🏁 Summary

> **EV-Range-Prediction-PINN**  
> blends **deep learning and vehicle physics**  
> to deliver **accurate, interpretable, and deployable** range estimation for the next generation of smart electric vehicles.
