# train.py
import json
import pickle
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Your details — CHANGE THESE
NAME = "S Srinath Bharadwaj"
REG_NO = "2022bcs0175"

# Load dataset
data = load_wine()
X, y = data.data, data.target.astype(float)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = Ridge()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R2:  {r2:.4f}")
print(f"Run completed by: {NAME} - {REG_NO}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metrics
metrics = {"mse": mse, "r2": r2}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("model.pkl and metrics.json saved.")
