import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
model = joblib.load("data/ids_model.pkl")
df = pd.read_csv("data/DrDoS_DNS.csv")
feature_names = df.drop('label', axis=1).columns
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importance Score")
plt.title("Feature Importance for DrDoS Detection")
plt.gca().invert_yaxis()
plt.show()
