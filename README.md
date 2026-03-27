# LNG Heat Exchanger Frost Monitoring System

A full-stack application for real-time monitoring and ML-based prediction of CO₂ freeze-out in shell-and-tube LNG heat exchangers. Combines first-principles physics simulation with machine learning to answer three critical operational questions.

> **Live Demo**: Run `docker compose up` and open [http://localhost:3000](http://localhost:3000)

---

## Business Objectives

| Question | Description | Approach |
|----------|-------------|----------|
| **A** | Are there unexpected correlations between input data and ΔP increase? | Feature importance analysis on trained Lasso model |
| **B** | Can existing sensors proxy for heavy hydrocarbon component loading? | Identify observable DCS/SCADA features selected by the ML pipeline |
| **C** | What is the expected runtime / defrost date? | Autoregressive dP_error trend extrapolation to forecast alarm crossing |

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend      │     │    Backend       │     │    MLflow        │
│  Vite + React    │────▶│  FastAPI         │────▶│  Model Serving   │
│  (nginx :3000)   │ SSE │  (:8000)         │HTTP │  (:5001)         │
│                  │◀────│                  │◀────│  Tracking (:5000)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │                        │
        │              ┌─────────┴─────────┐     ┌───────┴────────┐
        │              │ data/simulated/   │     │ mlartifacts/    │
        └──────────────│ combined_dataset  │     │ models + figs   │
                       └───────────────────┘     └────────────────┘
```

| Service | Container | Port | Technology |
|---------|-----------|------|------------|
| **Frontend** | `hx-frontend` | 3000 | Vite + React 18 + TypeScript + Recharts, served via nginx |
| **Backend** | `hx-backend` | 8000 | FastAPI + SSE streaming + httpx for model calls |
| **MLflow** | `hx-mlflow` | 5000/5001 | MLflow Tracking Server (5000) + Model Serving (5001) |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) Python 3.11+ for local training

### Run the Application

```bash
# Start all services (builds containers on first run)
docker compose up --build

# Access the UI
open http://localhost:3000

# Access MLflow tracking UI
open http://localhost:5000
```

### Train Models (Optional — pre-trained models are included)

```bash
# Create virtual environment
python -m venv hx_frost
source hx_frost/Scripts/activate  # Windows Git Bash
# source hx_frost/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate simulation data (if not present)
python scripts/generate_dataset.py

# Run EDA
python eda/eda_analysis.py

# Train all 7 models with Optuna optimization
python training/train.py

# Train specific models
python training/train.py --models lasso ridge random_forest
```

---

## Application Features

### Monitoring Page (`/`)

- **Interactive HX Diagram**: SVG visualization of a shell-and-tube heat exchanger with real-time sensor readings (T_h,in, T_h,out, T_c,in, T_c,out, ΔP) and frost visualization
- **Scenario Selector**: Dropdown with 5 simulation scenarios (normal, gradual freezing, rapid freezing, defrost recovery, partial blockage)
- **Real-Time Streaming**: SSE-based data streaming at 1 point/second, simulating historian data
- **ML Inference**: Each data point sent to MLflow-served Lasso model for dP_error prediction
- **Alert System**: Visual alerts when predicted ΔP exceeds alarm threshold (943.3 Pa)
- **Runtime Forecast**: Real-time estimate of remaining hours until defrost is required
- **Live Charts**: ΔP trend (actual vs predicted) and temperature evolution via Recharts

### Interpretability Page (`/interpretability`)

- **Question A — Feature Importance**: Bar chart showing Lasso model coefficients, color-coded by signal type (observable vs derived)
- **Question B — Proxy Features**: Analysis of which DCS-available sensors correlate with heavy hydrocarbon loading
- **Question C — Runtime Forecast**: Defrost date projection with trend extrapolation methodology
- **Model Comparison**: All 7 trained models with their top features and forecast data
- **MLflow Artifacts**: Training-time figures (residuals, importance, forecast) served from artifact store

---

## Hybrid PE-ML Methodology

The system uses a **hybrid first-principles + machine learning** approach:

$$\Delta P_{\text{pred}}(t) = \Delta P_{\text{ideal}} + \Delta P_{\text{error,ML}}(t)$$

- **First-principles baseline** ($\Delta P_{\text{ideal}}$): Constant clean-tube ΔP computed from Dittus-Boelter, Kern, and Churchill correlations
- **ML error model** ($\Delta P_{\text{error,ML}}$): Trained to predict the deviation from baseline caused by frost accumulation
- **Interpretability**: Each model coefficient has direct physical meaning

### ML Pipeline

```
Raw Data → Error Computation → Lag Features (t-1..t-3) → Rate-of-Change
         → Yeo-Johnson Transform (λ=0.61) → StandardScaler
         → SelectKBest(f_regression) → RFE (linear models)
         → Estimator → Inverse Transform → Physical Clipping
```

### Trained Models (Best → Worst by Test R²)

| Model | Test R² | Test MAE [Pa] |
|-------|---------|---------------|
| **Lasso** (production) | **0.9697** | — |
| Ridge | 0.9630 | — |
| Linear Regression | 0.9606 | — |
| Random Forest | 0.9525 | — |
| Gradient Boosting | 0.9461 | — |
| Elastic Net | 0.9194 | — |
| SVR | 0.7490 | — |

### Cross-Validation Strategy

- **Temporal group split**: Train on runs 1+2, test on run 3 (no time leakage)
- **Inner CV**: TimeSeriesSplit (n_splits=5) for Optuna tuning
- **Optuna**: 80 trials per model, 5-minute timeout, TPE sampler, median pruner

---

## Physical Model

### CO₂ Sublimation Equilibrium

The frost point is determined by the Antoine-type Clausius-Clapeyron fit to NIST solid-vapour data (Span & Wagner 1996), valid over 154–216.58 K:

$$\ln\!\left(\frac{P_{\mathrm{sub}}}{\mathrm{Pa}}\right) = 27.630 - \frac{3134.4}{T\,[\mathrm{K}]}$$

Deposition occurs when the bulk CO₂ mole fraction exceeds the equilibrium value at the wall:

$$y_{\mathrm{eq}}(T_w) = \frac{P_{\mathrm{sub}}(T_w)}{P_{\mathrm{total}}}$$

### PDE System — Method of Lines on $z \in [0, L]$, $t \in [0, T_{\mathrm{end}}]$

Three coupled equations are solved at $N = 100$ axial nodes.

**Tube-side gas** (flowing in $+z$ direction):

$$\frac{\partial T_h}{\partial t} + u_h(z,t)\,\frac{\partial T_h}{\partial z} = -\frac{U(z,t)\cdot 4}{D_h(z,t)\,\rho_h\, c_{p,h}}\,(T_h - T_c)$$

**Shell-side LNG** (counter-flow, flowing in $-z$ direction):

$$\frac{\partial T_c}{\partial t} - u_c\,\frac{\partial T_c}{\partial z} = +\frac{U(z,t)\cdot N_t\,\pi\, D_h(z,t)}{\rho_c\, c_{p,c}\, A_{sh}}\,(T_h - T_c)$$

**CO₂ frost layer** (Stefan-type deposition / erosion; Maqsood et al. 2014):

$$\frac{\partial \delta_f}{\partial t} = k_{\mathrm{dep}}\,\max\!\bigl(0,\; y_{\mathrm{CO_2}} - y_{\mathrm{eq}}(T_w)\bigr) \;-\; k_{\mathrm{rem}}\,u_h(z,t)\,\delta_f$$

where the frost surface temperature $T_w$ is obtained from the thermal resistance network:

$$T_w = T_h - \frac{T_h - T_c}{R_\mathrm{total}}\cdot\frac{1}{h_h}, \qquad R_\mathrm{total} = \frac{1}{h_h} + \frac{\delta_f}{k_{\mathrm{CO_2}}} + \frac{t_w}{k_w} + \frac{1}{h_c}$$

### Heat Transfer Correlations

**Tube-side — Dittus-Boelter (1930):** (gas cooled, $n=0.3$)

$$Nu_h = 0.023\,Re_h^{0.8}\,Pr_h^{0.3}, \qquad Re_h = \frac{G_h\,D_h}{\mu_h}, \qquad G_h = \frac{\dot{m}_h}{A_{\mathrm{flow}}(z,t)}$$

A piecewise fallback handles laminar ($Nu=3.66$) and transitional regimes.

**Shell-side — Kern (1950):**

$$Nu_c = 0.36\,Re_s^{0.55}\,Pr_c^{1/3}, \qquad D_e = \frac{4\!\left(\dfrac{\sqrt{3}}{4}p^2 - \dfrac{\pi}{8}d_o^2\right)}{\dfrac{\pi}{2}d_o} \quad \text{(triangular pitch)}$$

**Overall heat transfer coefficient:**

$$\frac{1}{U(z,t)} = \frac{1}{h_h(z,t)} + \frac{\delta_f(z,t)}{k_{\mathrm{CO_2}}} + \frac{t_w}{k_w} + \frac{1}{h_c}$$

### Pressure Drop

**Tube-side — Darcy-Weisbach with Churchill (1977) friction factor:**

$$\frac{\mathrm{d}P}{\mathrm{d}z} = \frac{f_D\,G_h(z,t)^2}{2\,\rho_h\,D_h(z,t)}, \qquad \Delta P(t) = \int_0^L \frac{\mathrm{d}P}{\mathrm{d}z}\,\mathrm{d}z$$

The Churchill (1977) friction factor is valid for **all** Reynolds numbers and wall roughness:

$$f_D = 8\left[\left(\frac{8}{Re}\right)^{12} + (A+B)^{-3/2}\right]^{1/12}$$

$$A = \left[2.457\ln\!\frac{1}{\left(7/Re\right)^{0.9} + 0.27\,(\varepsilon/D)}\right]^{16}, \qquad B = \left(\frac{37530}{Re}\right)^{16}$$

**Shell-side — Kern (1950):**

$$\Delta P_s = \frac{f_s\,G_s^2\,D_s\,(N_b+1)}{2\,\rho_c\,D_e}, \qquad f_s = e^{0.576 - 0.19\ln Re_s}$$

### Frost-driven ΔP nonlinearity

As $\delta_f$ grows, the mass flux through the tube increases at constant $\dot{m}_h$:

$$G_h(z,t) = \frac{\dot{m}_h}{N_t\,\frac{\pi}{4}\,D_h(z,t)^2}, \qquad \Delta P \propto \frac{G_h^2}{D_h} \propto \frac{1}{D_h^5}$$

A 5 % reduction in $D_h$ produces approximately **28 % increase in ΔP**, making it a highly sensitive early-warning signal.

---

## Numerical Method

| Property | Value |
|---|---|
| Spatial nodes $N$ | 100 |
| Spatial scheme | First-order upwind (MOL) |
| Temporal integrator | `scipy.solve_ivp`, method `Radau` (stiff, A-stable) |
| Max time step | 60 s |
| Relative tolerance | $10^{-4}$ |
| Absolute tolerance | $10^{-7}$ |

---

## Operating Scenarios

| Scenario | Duration | CO₂ [mol%] | ΔP change | Description |
|---|---|---|---|---|
| `normal_operation` | 6 h | 0.5 | ~3 % | Stable, below frost point |
| `gradual_freezing` | 12 h | 2.0 | 20–40 % | Slow frost build-up |
| `rapid_freezing` | 4 h | 3.0 | **65 %** | Alarm crossed in <4 h |
| `defrost_recovery` | 2 h | 0.5 | −40 % | Warm purge, frost retreats |
| `partial_blockage` | 8 h | 2.5 | 30–50 % | Spatially non-uniform (5× k_dep at cold inlet) |

### Sensor Noise Model

Gaussian white noise is added to all measurable channels to simulate SCADA/DCS instrumentation:

| Channel | Noise level | Instrument class |
|---|---|---|
| Temperature (RTD) | $\sigma_T = 0.5$ K | IEC 60751 Class A |
| Pressure drop | $\sigma_{\Delta P} = 0.2\%$ of reading | Smart transmitter |
| Mass flow | $\sigma_{\dot{m}} = 0.5\%$ of reading | Coriolis |

---

## Repository Structure

```
heat_exchanger_monitoring/
├── docker-compose.yml           Orchestration for all 3 containers
│
├── backend/                     FastAPI backend
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py              Application entry point + CORS + lifespan
│       ├── config.py            Environment-based settings
│       ├── routes/
│       │   ├── scenarios.py     SSE streaming + scenario listing
│       │   └── interpretability.py  ML artifacts + figure serving
│       └── services/
│           ├── data_service.py  Data loading + feature engineering
│           └── model_service.py MLflow model proxy + artifact reader
│
├── frontend/                    React frontend
│   ├── Dockerfile               Multi-stage (node build → nginx)
│   ├── nginx.conf               SPA routing + API proxy
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx              Router + layout
│       ├── api.ts               Backend API client
│       ├── types.ts             TypeScript interfaces
│       ├── index.css            LNG blue/white theme (CSS variables)
│       ├── pages/
│       │   ├── MainPage.tsx     Monitoring dashboard
│       │   └── InterpretabilityPage.tsx  Business questions A/B/C
│       └── components/
│           ├── HeatExchangerDiagram.tsx  SVG shell-and-tube HX
│           ├── StreamingChart.tsx        ΔP + temperature charts
│           ├── AlertPanel.tsx           Alarm banners
│           ├── MetricsPanel.tsx         Live process metrics
│           └── RuntimeForecast.tsx      Defrost time estimate
│
├── mlflow/                      MLflow serving container
│   ├── Dockerfile
│   └── entrypoint.sh           Tracking server + model serving
│
├── src/                         Physics simulation engine
│   ├── correlations.py          HT & pressure drop correlations
│   ├── freezing_model.py        CO₂ sublimation, frost growth
│   ├── heat_exchanger.py        Method-of-Lines PDE solver
│   └── scenarios.py             Scenario configs + noise injection
│
├── training/                    ML training pipeline
│   ├── train.py                 Main entry point (Optuna + MLflow)
│   ├── config/                  YAML experiment/feature/model configs
│   └── modules/
│       ├── data/                Loader + preprocessor
│       ├── evaluation/          Metrics + interpretability reporters
│       ├── models/              Model registry + pipeline builder
│       └── transformers/        Yeo-Johnson, lags, rate-of-change
│
├── eda/                         Exploratory data analysis
│   ├── eda_analysis.py
│   ├── EDA_REPORT.md
│   └── enriched_dataset.csv     Dataset with ideal/error features
│
├── data/simulated/              Generated simulation data
│   ├── combined_dataset.csv     10,440 rows (5 scenarios × 3 runs)
│   └── [per-scenario CSVs]
│
├── mlartifacts/                 MLflow experiment artifacts
│   └── 1/                       7 model runs + registered models
│
├── scripts/
│   ├── simulate.py              Single-scenario simulation
│   └── generate_dataset.py      Full dataset generation
│
├── requirements.txt             Unified Python dependencies
└── README.md
```

---

## Data Pipeline

### Simulation

```bash
python scripts/generate_dataset.py
```

### Output Files

```
data/simulated/
├── normal_operation.csv
├── gradual_freezing.csv
├── rapid_freezing.csv
├── defrost_recovery.csv
├── partial_blockage.csv
└── combined_dataset.csv
```

---

## Default STHE Geometry

| Parameter | Symbol | Value |
|---|---|---|
| Tube length | $L$ | 6 m |
| Tube inner diameter (clean) | $D_{h,0}$ | 20 mm |
| Number of tubes | $N_t$ | 200 |
| Tube pitch (triangular) | $p$ | 32 mm |
| Shell inner diameter | $D_s$ | 600 mm |
| Number of baffles | $N_b$ | 12 |

---

## Validated Physics Outputs

At nominal conditions ($\dot{m}_h = 5$ kg/s, $T_{h,\mathrm{in}} = 250$ K, $T_{c,\mathrm{in}} = 120$ K, $P = 40$ bar):

- **NTU** = $U \bar{A} / C_{\min} \approx 1.67$ → effectiveness $\varepsilon \approx 0.69$
- $T_{h,\mathrm{out}} \approx 161$ K (gas cooled ~89 K) ✓
- $T_{c,\mathrm{out}} \approx 181$ K (LNG heated ~61 K) ✓
- Clean-baseline $\Delta P \approx 619$ Pa ✓
- $h_{\mathrm{shell}} \approx 619$ W/m²K, $U \approx 255$ W/m²K ✓

---

## References

1. Maqsood, K. et al. (2014). Cryogenic packed beds for CO₂ capture. *Chemical Engineering Journal*, 253, 327–336.
2. Bai, F. & Newell, T.A. (2002). Modeling of CO₂ freezing in a cryogenic heat exchanger. *Int. J. Refrigeration*, 25(4), 476–484.
3. Churchill, S.W. (1977). Friction-factor equation spans all fluid-flow regimes. *Chemical Engineering*, 84(24), 91–92.
4. Dittus, F.W. & Boelter, L.M.K. (1930). Heat transfer in automobile radiators. *Univ. California Pub. Eng.*, 2, 443.
5. Kern, D.Q. (1950). *Process Heat Transfer*. McGraw-Hill.
6. Span, R. & Wagner, W. (1996). A new equation of state for CO₂. *J. Phys. Chem. Ref. Data*, 25(6), 1509–1596.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend health check |
| GET | `/api/scenarios` | List scenarios with metadata |
| GET | `/api/stream/{scenario}?run_id=1` | SSE stream of data points with ML predictions |
| GET | `/api/interpretability` | Feature importance, forecast, figures for best model |
| GET | `/api/interpretability/figures/{filename}` | Serve PNG artifact |
| GET | `/api/models` | All trained models comparison data |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Simulation** | Python, SciPy (Radau), NumPy | Physics-based PDE solver for heat/mass transfer |
| **ML Training** | scikit-learn, Optuna, MLflow | Automated hyperparameter optimization + experiment tracking |
| **Model Serving** | MLflow Models | REST API model inference (production MLOps pattern) |
| **Backend** | FastAPI, SSE-Starlette, httpx | Async API with Server-Sent Events for real-time streaming |
| **Frontend** | React 18, TypeScript, Recharts | Interactive monitoring dashboard with live charts |
| **Production** | nginx, Docker Compose | Multi-container orchestration with health checks |

---

## License

This project is a demonstration / proof-of-concept for LNG plant monitoring.