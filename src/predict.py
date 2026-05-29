import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

sample = pd.DataFrame({
    "amount": [4500],
    "transactions_today": [15],
    "distance_from_home": [800],
    "international": [1]
})

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Fraud Transaction Detected")
else:
    print("Legitimate Transaction")