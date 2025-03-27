import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')  # Handle encoding issues
    return df

def preprocess_data(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    return df

def create_rfm_features(df):
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})
    return rfm

def scale_features(rfm):
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm)
    return scaled_rfm, scaler

def apply_kmeans(scaled_rfm, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_rfm)
    return clusters, kmeans

def save_processed_data(rfm, clusters, output_path):
    rfm['Cluster'] = clusters
    rfm.to_csv(output_path, index=True)
    return rfm

if __name__ == "__main__":
    file_path = "data.csv"  # Ensure the dataset is placed in the working directory
    df = load_data(file_path)
    df = preprocess_data(df)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    rfm = create_rfm_features(df)
    scaled_rfm, scaler = scale_features(rfm)
    clusters, kmeans = apply_kmeans(scaled_rfm, n_clusters=4)
    processed_data = save_processed_data(rfm, clusters, "processed_rfm.csv")
    print("Customer segmentation completed and saved as processed_rfm.csv")
