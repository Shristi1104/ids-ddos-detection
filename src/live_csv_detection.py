import pandas as pd
import numpy as np
import joblib
model = joblib.load("data/ids_model.pkl")
scaler = joblib.load("data/scaler.pkl")
def detect_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    if 'label' in df.columns:
        df = df.drop('label', axis=1)

    scaled_data = scaler.transform(df.values)
    predictions = model.predict(scaled_data)

    df['Prediction'] = np.where(predictions == 1, "DrDoS_DNS", "BENIGN")
    return df
result = detect_from_csv("data/DrDoS_DNS.csv")
print(result[['Prediction']].value_counts())
