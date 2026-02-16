import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
print("Training data loaded")
ids_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
ids_model.fit(X_train, y_train)
print("Final IDS model trained")
joblib.dump(ids_model, "data/ids_model.pkl")
print("Model saved as ids_model.pkl")
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, "data/scaler.pkl")
print("Scaler saved as scaler.pkl")
def predict_traffic(sample):
    """
    sample: list or numpy array of 15 feature values
    """
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    prediction = ids_model.predict(sample)

    if prediction[0] == 0:
        return "BENIGN"
    else:
        return "DrDoS_DNS ATTACK"
example_sample = X_train[0]
result = predict_traffic(example_sample)
print("Prediction for sample traffic:", result)
