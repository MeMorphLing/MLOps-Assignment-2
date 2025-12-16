from __future__ import annotations
import pandas as pd
import pickle
import os
from typing import cast, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Data
print("Loading data...")
# Explicitly tell Pylance this is a DataFrame
df = cast(pd.DataFrame, pd.read_csv('data/housing.csv'))

X = df.drop(columns=['MedHouseVal'])  # Features
y = df['MedHouseVal']                 # Target

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Model MSE: {mse}")

# 5. Save Model
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to models/model.pkl")