import numpy as np
import pandas as pd

def generate_data(n=1000, devices=5):
    all_data = []

    for d in range(devices):
        temp = 40 + np.random.normal(0, 1, n)
        vibration = 0.02 + np.random.normal(0, 0.002, n)
        voltage = 12 + np.random.normal(0, 0.1, n)
        noise = np.random.normal(0, 0.005, n)

        packet_loss = np.random.uniform(0, 0.5, n)
        latency = np.random.uniform(1, 5, n)
        link_errors = np.zeros(n)

        start = np.random.randint(600, 800)

        # 🔥 Thermal degradation
        temp[start:] += np.linspace(0, 10, n-start)

        # ⚙ Mechanical wear
        vibration[start:] += np.linspace(0, 0.05, n-start)

        # ⚡ Electrical instability
        voltage[start:] += np.random.normal(0, 0.5, n-start)

        # 🌐 Network degradation (CORRELATED WITH TEMP)
        packet_loss[start:] += (temp[start:] - 45) * 0.05
        latency[start:] += (temp[start:] - 45) * 0.2

        # Random link errors after degradation
        link_errors[start:] = np.random.randint(0, 2, n-start)

        df = pd.DataFrame({
            "device_id": d,
            "temp": temp,
            "vibration": vibration,
            "voltage": voltage,
            "noise": noise,
            "packet_loss": packet_loss,
            "latency": latency,
            "link_errors": link_errors
        })

        all_data.append(df)

    final_df = pd.concat(all_data)
    final_df.to_csv("data/raw/data.csv", index=False)

if __name__ == "__main__":
    generate_data()