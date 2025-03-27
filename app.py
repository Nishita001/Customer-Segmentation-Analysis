import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_segmented_data():
    return pd.read_csv("processed_rfm.csv", index_col=0)

def visualize_clusters(df):
    st.subheader("Customer Segments Visualization")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.boxplot(x='Cluster', y='Recency', data=df, ax=axes[0])
    axes[0].set_title("Recency by Cluster")
    
    sns.boxplot(x='Cluster', y='Frequency', data=df, ax=axes[1])
    axes[1].set_title("Frequency by Cluster")
    
    sns.boxplot(x='Cluster', y='Monetary', data=df, ax=axes[2])
    axes[2].set_title("Monetary by Cluster")
    
    st.pyplot(fig)

def main():
    st.title("Customer Segmentation Dashboard")
    df = load_segmented_data()
    st.dataframe(df.head())
    visualize_clusters(df)

if __name__ == "__main__":
    main()
