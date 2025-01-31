import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from zenml import pipeline
from zenml.steps import step

# Define the pipeline steps

@step
def load_df(url: str) -> pd.DataFrame:
    """Load CSV from URL."""
    df = pd.read_csv(url)
    return df

@step
def filter_df(df: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
    """Filter the DataFrame to keep only the selected columns."""
    df_filtered = df[selected_columns]
    return df_filtered

@step
def apply_kmeans_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Apply K-Means clustering on Returns and Volatility columns."""
    X = df[['Returns', 'Volatility']]

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Plotting the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Returns'], df['Volatility'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('Returns')
    plt.ylabel('Volatility')
    plt.title('K-Means Clustering on Returns and Volatility')
    plt.colorbar(label='Cluster')
    plt.show()

    return df

# Define the pipeline

@pipeline
def investment_pipeline(url: str, selected_columns: list, n_clusters: int = 3):
    df = load_df(url)
    filtered_df = filter_df(df, selected_columns)
    clustered_df = apply_kmeans_clustering(filtered_df, n_clusters)
    return clustered_df

# Set the URL and selected columns for your experiment

url = "https://raw.githubusercontent.com/sashaTen/investment_app/refs/heads/main/invest/S%26P500data/clusters_df.csv"
selected_columns = [
    "Ticker", "Returns", "Volatility", "MarketCap", 
    "ProfitMargins", "DebtToEquity", "ReturnOnEquity", "sharpe_ratio"
]

# Run the pipeline
if __name__ == "__main__":
    # Execute the ZenML pipeline
    df_clustered = investment_pipeline(url=url, selected_columns=selected_columns, n_clusters=3)
    print(df_clustered)
