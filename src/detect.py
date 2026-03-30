import pandas as pd
import joblib
import numpy as np
from logger import log_data

def detect():
    df = pd.read_csv("data/processed/features.csv")

    model = joblib.load("data/models/model.pkl")
    scaler = joblib.load("data/models/scaler.pkl")

    X = scaler.transform(df.drop(columns=["device_id"]))

    df["score"] = model.decision_function(X)
    df["anomaly"] = model.predict(X)

    # 🎯 HEALTH SCORE (balanced)
    df["health_score"] = 100 \
        - df["temp"] * 0.4 \
        - df["vibration"] * 400 \
        - df["voltage_std"] * 10 \
        - df["network_score"] * 2

    df["health_score"] = df["health_score"].clip(0, 100)

    # 📉 RUL (trend-based)
    df["temp_rate"] = df.groupby("device_id")["temp"].diff().rolling(20).mean()

    df["rul"] = (55 - df["temp"]) / (df["temp_rate"] + 1e-5)
    df["rul"] = df["rul"].clip(0)

    # ⚠️ FAILURE RISK
    df["failure_risk"] = 1 / (df["rul"] + 1)

    df["failure_label"] = "Stable"
    df.loc[df["failure_risk"] > 0.03, "failure_label"] = "Degrading"
    df.loc[df["failure_risk"] > 0.07, "failure_label"] = "Failure Soon"

    df = df.fillna(0)

    log_data(df)
    df.to_csv("data/processed/results.csv", index=False)

    print("✅ Detection Complete (AI Enhanced)")

if __name__ == "__main__":
    detect()