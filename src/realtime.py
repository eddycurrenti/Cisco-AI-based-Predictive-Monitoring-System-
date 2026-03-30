import time
import pandas as pd
import joblib

df = pd.read_csv("data/processed/features.csv")

model = joblib.load("data/models/model.pkl")
scaler = joblib.load("data/models/scaler.pkl")

print("🚀 Real-Time Monitoring Started")

for i in range(len(df)):
    sample = df.iloc[i:i+1]

    X = scaler.transform(sample.drop(columns=["device_id"]))
    score = model.decision_function(X)[0]

    if score < -0.05:
        status = "🔴 CRITICAL FAILURE"
    elif score < -0.01:
        status = "🟠 EARLY WARNING"
    else:
        status = "🟢 NORMAL"

    print(f"[{i}] Score: {round(score,4)} → {status}")

    time.sleep(0.2)