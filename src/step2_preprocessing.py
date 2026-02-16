import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/DrDoS_DNS.csv")
print("Dataset loaded for preprocessing")
X = df.drop('label', axis=1)
y = df['label']
print("Features shape:", X.shape)
print("Label shape:", y.shape, "\n")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("label encoding completed")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))), "\n")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(" train-test split completed")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape, "\n")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed\n")
np.save("data/X_train.npy", X_train_scaled)
np.save("data/X_test.npy", X_test_scaled)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)
print("Processed data saved successfully")

import joblib
joblib.dump(scaler, "data/scaler.pkl")
print("scaler saved correctly during preprocessing")