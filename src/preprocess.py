import pandas as pd

def preprocess():
    df = pd.read_csv("data/raw/data.csv")

    df_smooth = df.groupby("device_id").rolling(10).mean().reset_index(drop=True)

    feat = pd.DataFrame()
    feat["device_id"] = df["device_id"]

    # Core signals
    feat["temp"] = df_smooth["temp"]
    feat["vibration"] = df_smooth["vibration"]
    feat["voltage"] = df_smooth["voltage"]

    # 🔥 Trend features (VERY IMPORTANT)
    feat["temp_trend"] = feat.groupby("device_id")["temp"].diff()
    feat["vib_trend"] = feat.groupby("device_id")["vibration"].diff()

    # ⚡ Instability
    feat["voltage_std"] = df.groupby("device_id")["voltage"].rolling(10).std().reset_index(drop=True)

    # 🌐 Network features
    feat["packet_loss"] = df["packet_loss"]
    feat["latency"] = df["latency"]
    feat["link_errors"] = df["link_errors"]

    # Network issue score
    feat["network_score"] = (
        feat["packet_loss"] * 2 +
        feat["latency"] * 0.5 +
        feat["link_errors"] * 5
    )

    feat = feat.dropna()

    feat.to_csv("data/processed/features.csv", index=False)

if __name__ == "__main__":
    preprocess()