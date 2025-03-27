import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("processed_rfm.csv")

# Selecting only RFM columns
X = df[['Recency', 'Frequency', 'Monetary']]

# Scaling the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a better K-Means model
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Save the model, scaler, and feature names
joblib.dump(kmeans, "rfm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")  # Save feature names

print("✅ Model, Scaler, and Feature Names saved successfully!")