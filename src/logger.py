import pandas as pd
from datetime import datetime

def log_data(df):
    df["timestamp"] = datetime.now()
    df.to_csv("data/processed/history.csv", mode='a', header=False, index=False)