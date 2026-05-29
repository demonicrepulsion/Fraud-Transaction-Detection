import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/fraud_data.csv")

X = df.drop("fraud", axis=1)
y = df["fraud"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "models/fraud_model.pkl")

print("Model saved successfully!")