# CO2 Emissions Prediction Model

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## 📌 Overview

A machine learning pipeline that predicts vehicular CO2 emissions for Canadian vehicles using advanced regression techniques. This project leverages scikit-learn's preprocessing and ensemble methods to build a robust predictive model based on vehicle characteristics.

**Key Objective**: Estimate CO2 emissions (g/km) with high accuracy using vehicle attributes such as engine size, fuel consumption, and vehicle classification.

---

## ✨ Features

- ✅ Automated feature preprocessing for numerical and categorical data
- ✅ Scalable data imputation strategies
- ✅ One-hot encoding for categorical variables
- ✅ Random Forest ensemble learning
- ✅ Comprehensive model evaluation metrics
- ✅ Serialized model persistence for deployment

---

## 📊 Dataset

| Attribute | Type | Description |
|-----------|------|-------------|
| **CO2 Emissions(g/km)** | Target | Vehicle CO2 emissions output |
| **Engine Size(L)** | Numerical | Engine displacement in liters |
| **Cylinders** | Numerical | Number of engine cylinders |
| **Fuel Consumption City** | Numerical | City fuel consumption (L/100 km) |
| **Fuel Consumption Hwy** | Numerical | Highway fuel consumption (L/100 km) |
| **Fuel Consumption Comb** | Numerical | Combined fuel consumption (L/100 km) |
| **Fuel Consumption Comb (mpg)** | Numerical | Combined fuel consumption (mpg) |
| **Make** | Categorical | Vehicle manufacturer |
| **Model** | Categorical | Vehicle model name |
| **Vehicle Class** | Categorical | Classification of vehicle type |
| **Transmission** | Categorical | Transmission type |
| **Fuel Type** | Categorical | Type of fuel used |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

Install required packages:

```bash
pip install pandas numpy scikit-learn joblib
```

### Quick Setup

```bash
git clone <repository-url>
cd co2-emissions-prediction
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
co2-emissions-prediction/
├── app.py                           # Main training script
├── emission_pipeline.pkl            # Trained model (generated)
├── CO2_Emissions_Canada.csv         # Dataset
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔧 Model Architecture

### Preprocessing Pipeline

```
Input Data
    ├── Numerical Branch
    │   ├── Imputation (Mean Strategy)
    │   └── Standardization (StandardScaler)
    │
    └── Categorical Branch
        ├── Imputation (Most Frequent Strategy)
        └── Encoding (OneHotEncoder)
            │
            └── Merged Features
                └── Random Forest Regressor
                    └── Predictions
```

### Pipeline Components

| Component | Method | Purpose |
|-----------|--------|---------|
| Numerical Imputation | Mean | Handle missing values in numerical features |
| Numerical Scaling | StandardScaler | Normalize feature ranges |
| Categorical Imputation | Most Frequent | Fill missing categorical values |
| Categorical Encoding | OneHotEncoder | Convert categories to numerical format |
| Regression Model | Random Forest | Ensemble learning for predictions |

---

## 📈 Usage

### Training the Model

Run the main script to train and evaluate the model:

```bash
python app.py
```

**Output**:
```
Model Metrics:
MSE: 9.77
RMSE: 3.13
MAE: 1.84
R2_Score: 0.9970
```

### Using the Trained Model

Load and make predictions on new data:

```python
import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load("emission_pipeline.pkl")

# Prepare new vehicle data
new_vehicles = pd.DataFrame({
    'Make': ['Toyota'],
    'Model': ['Camry'],
    'Vehicle Class': ['Mid-size'],
    'Transmission': ['Automatic'],
    'Fuel Type': ['Regular Gasoline'],
    'Engine Size(L)': [2.5],
    'Cylinders': [4],
    'Fuel Consumption City (L/100 km)': [9.8],
    'Fuel Consumption Hwy (L/100 km)': [7.5],
    'Fuel Consumption Comb (L/100 km)': [8.8],
    'Fuel Consumption Comb (mpg)': [32.1]
})

# Generate predictions
predictions = pipeline.predict(new_vehicles)
print(f"Predicted CO2 Emissions: {predictions[0]:.2f} g/km")
```

---

## 📊 Performance Metrics

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MSE** | Σ(y_true - y_pred)² / n | Average squared prediction error |
| **RMSE** | √MSE | Root mean squared error in original units |
| **MAE** | Σ\|y_true - y_pred\| / n | Mean absolute deviation |
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0-1) |

### Model Performance Criteria

- **Excellent**: R² > 0.85
- **Good**: 0.70 < R² ≤ 0.85
- **Fair**: 0.50 < R² ≤ 0.70
- **Poor**: R² ≤ 0.50

---

## 🔍 Data Processing Details

### Train-Test Split
- Training Set: 70%
- Testing Set: 30%
- Random State: Not fixed (varies across runs)

### Feature Handling

**Numerical Features** (6 features):
- Missing values imputed with column mean
- Scaled using Z-score normalization

**Categorical Features** (5 features):
- Missing values imputed with most frequent value
- Encoded using one-hot encoding with unknown value handling

---

## 🛠️ Advanced Configuration

### Customizing the Model

```python
from sklearn.ensemble import RandomForestRegressor

# Modify Random Forest hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

# Update pipeline
pipeline.named_steps['model'] = rf_model
pipeline.fit(X_train, y_train)
```

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 1.3.0 | Data manipulation and analysis |
| numpy | ≥ 1.21.0 | Numerical computations |
| scikit-learn | ≥ 1.0.0 | Machine learning algorithms |
| joblib | ≥ 1.0.0 | Model serialization |

---

## 🚀 Future Enhancements

- [ ] Hyperparameter optimization with GridSearchCV
- [ ] K-fold cross-validation implementation
- [ ] Feature importance analysis and visualization
- [ ] Model comparison with Gradient Boosting and XGBoost
- [ ] Input validation and error handling
- [ ] REST API deployment with Flask/FastAPI
- [ ] Docker containerization
- [ ] Automated retraining pipeline

---

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## ✉️ Contact & Support

For questions, issues, or contributions, please contact the development team or submit an issue on the project repository.

---

<div align="center">

**Last Updated**: November 2025

Made with ❤️ for environmental sustainability

</div>
