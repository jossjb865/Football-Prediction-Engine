# ⚽ Football Prediction Engine (Liga MX)

An advanced machine learning-powered prediction engine tailored for **Liga MX**, leveraging a sophisticated two-level stacking ensemble architecture, multi-source data processing, and automated CI/CD pipelines via GitHub Actions.

---

## 🌟 Key Features

- **Multi-Model Level-0 Stack**: Combines state-of-the-art predictive algorithms including **XGBoost**, **CatBoost**, **Deep Neural Networks (DNN)**, and **Dixon-Coles Poisson statistical modeling**.
- **Meta-Model Level-1 Stacking**: Uses TimeSeriesSplit cross-validation and Logistic Regression to intelligently aggregate base model predictions and output robust match outcome probabilities (Home Win, Draw, Away Win).
- **Automated Workflows**: Fully automated daily prediction pipelines and weekly model retraining powered by GitHub Actions.
- **iSportsAPI Integration**: Real-time fixture fetching, historical performance data harvesting, and clean feature engineering.

---

## 📂 Project Architecture

```text
Football-Prediction-Engine/
├── .github/
│   └── workflows/
│       └── pipeline.yml       # Unified CI/CD training and prediction pipeline
├── artifacts/
│   └── models/                # Saved joblib ensemble models
├── config/
│   ├── logging_config.py      # Structured logging setup
│   └── settings.py            # Environment validation and configuration
├── ensemble/
│   └── super_stacking.py      # Two-level stacking ensemble implementation
├── models/                    # Individual base models (XGBoost, CatBoost, DNN, Dixon-Coles)
├── pipelines/                 # Training and prediction data pipelines
├── main.py                    # Entry point for training models
├── predict.py                 # Entry point for generating daily predictions
└── requirements.txt           # Project dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jossjb865/Football-Prediction-Engine.git
   cd Football-Prediction-Engine
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file or export the required keys in your shell:
   ```bash
   export ISPORTS_API_KEY="your_isports_api_key_here"
   export LIGA_MX_LEAGUE_ID="your_liga_mx_league_id_here"
   ```

---

## 🚀 Usage

### 1. Training the Models
To run the full training pipeline locally starting from a specific season year:
```bash
python main.py --min-year 2020
```

### 2. Generating Predictions
To generate match predictions using the trained ensemble:
```bash
python predict.py
```

---

## 🤖 GitHub Actions Automation

The repository uses a single unified workflow located in `.github/workflows/pipeline.yml` that handles:
- **Scheduled Triggers**: Runs automatically every day at 14:00 UTC.
- **Manual Dispatches**: Can be triggered manually anytime via the GitHub Actions tab (`workflow_dispatch`).
- **Artifact Generation**: Outputs daily prediction results as downloadable CSV files (`daily-predictions`).

---

## 🛡️ Requirements & Compatibility

- **Python**: `3.11+`
- **Libraries**: `scikit-learn`, `xgboost`, `catboost`, `pandas`, `numpy`, `joblib`, `torch` (for DNN)

---

## 📝 License

This project is licensed under the MIT License. Feel free to use, modify, and build upon it.
