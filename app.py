import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Check dataset
df = pd.read_csv("CO2 Emissions_Canada.csv")
print(df.head())
df.info()

# Seperate target and feature columns
X = df.drop(columns=["CO2 Emissions(g/km)"], axis=1)
y = df["CO2 Emissions(g/km)"]
X.info()
y.info()

# Divide numerical and categorical columns
numerical_cols = ["Engine Size(L)", "Cylinders", "Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)", "Fuel Consumption Comb (mpg)"]
categorical_cols = ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]

# Numerical imputation and standardization
numerical_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical imputation and encoding
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Joining the numerical and categorical pipelines
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Preprocessing and model selection pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])

# Dividing into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fitting the model
pipeline.fit(X_train, y_train)

# Prediction
prediction = pipeline.predict(X_test)

# Metrics/ model evaluation
mse = mean_squared_error(y_test ,prediction)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print(f"Model Metrics:\n"
      f"MSE: {mse}\n"
      f"RMSE: {rmse}\n"
      f"MAE: {mae}\n"
      f"R2_Score: {r2}")

# Model Saved
joblib.dump(pipeline, "emission_pipeline.pkl")

