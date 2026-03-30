import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib

def train():
    df = pd.read_csv("data/processed/features.csv")

    X = df.drop(columns=["device_id"])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_scaled)

    joblib.dump(model, "data/models/model.pkl")
    joblib.dump(scaler, "data/models/scaler.pkl")

    print("✅ Model trained with advanced features")

if __name__ == "__main__":
    train()