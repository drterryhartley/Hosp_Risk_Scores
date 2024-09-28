import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Generate Sample Data
np.random.seed(42)  # For reproducibility

# Number of sample hospitals
num_hospitals = 100

# Simulates data that would be replaced by RWE
data = {
    'Hospital_Name': [f'Hospital_{i}' for i in range(num_hospitals)],
    'Patient_Volume': np.random.randint(5000, 50000, num_hospitals),
    'Mortality_Rate': np.random.uniform(1, 10, num_hospitals),
    'Staffing_Ratio': np.random.uniform(0.5, 3.0, num_hospitals),
    'ER_Wait_Time': np.random.uniform(15, 300, num_hospitals),
    'Readmission_Rate': np.random.uniform(5, 30, num_hospitals),
    'Current_Funding_Level': np.random.uniform(10, 100, num_hospitals),
    'Location_Poverty_Index': np.random.uniform(0, 1, num_hospitals),
    'Risk_Score': np.random.uniform(0, 100, num_hospitals)  # Simulating risk score for this example
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess Data (remove categorical columns and handle missing values if any)
X = df.drop(columns=['Hospital_Name', 'Risk_Score'])  # Features
y = df['Risk_Score']  # Target

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training (Random Forest Regressor for Risk Scoring) RandomForrest is used due to the expectation of a non-linear relationship.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Step 6: Evaluate Model Performance - the root mean squared error (RMSE), which helps understand how far off the predictions are from the actual values
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Root Mean Squared Error: {rmse:.2f}')

# Step 7: Feature Importance (understanding which features impact the risk score the most)
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Step 8: Save the Model (Optional)
import joblib
joblib.dump(model, 'hospital_risk_scoring_model.pkl')

# Display the sample data and the results
import ace_tools as tools; tools.display_dataframe_to_user(name="Sample Hospital Data", dataframe=df)
