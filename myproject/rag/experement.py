
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/sashaTen/investment_app/refs/heads/main/invest/S%26P500data/clusters_df.csv"

selected_columns = [
        "Ticker", "Returns", "Volatility", "MarketCap", 
        "ProfitMargins", "DebtToEquity", "ReturnOnEquity", "sharpe_ratio"
    ]

def load_df(url):
    df = pd.read_csv(url)
    return df

def filter_df(df , selected_columns):
     df_filtered = df[selected_columns]
     return df_filtered


def apply_kmeans_clustering(df, n_clusters=3):
   
    X = df[['Returns', 'Volatility']]

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Display the DataFrame with the cluster labels
    print(df[['Ticker', 'Returns', 'Volatility', 'Cluster']])

    # Plotting the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Returns'], df['Volatility'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('Returns')
    plt.ylabel('Volatility')
    plt.title('K-Means Clustering on Returns and Volatility')
    plt.colorbar(label='Cluster')
    plt.show()

    return df
 # Load your filtered data
df = load_df(url)
filtered_df = filter_df(df , selected_columns)
df_clustered = apply_kmeans_clustering(filtered_df, n_clusters=3)
print(df_clustered.head())

