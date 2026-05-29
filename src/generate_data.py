import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

amount = np.random.uniform(10, 5000, n)
transactions_today = np.random.randint(1, 20, n)
distance_from_home = np.random.uniform(0, 1000, n)
international = np.random.randint(0, 2, n)

fraud = (
    (amount > 3000)
    & (distance_from_home > 500)
    & (international == 1)
).astype(int)

df = pd.DataFrame({
    "amount": amount,
    "transactions_today": transactions_today,
    "distance_from_home": distance_from_home,
    "international": international,
    "fraud": fraud
})

df.to_csv("data/fraud_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())