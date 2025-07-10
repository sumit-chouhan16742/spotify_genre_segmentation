# Spotify Songs' Genre Segmentation Project
# Submitted for AI Internship at Corizo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Step 1: Load the dataset
df = pd.read_csv("spotify dataset.csv")

# Step 2: Drop rows with missing key song information
df.dropna(subset=["track_name", "track_artist", "track_album_name"], inplace=True)

# Step 3: Select numeric audio features for analysis & clustering
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms'
]
X = df[features]

# Step 4: Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Audio Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")  # Saves the plot
plt.close()

# Step 6: Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 7: Perform KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Step 8: Add cluster and PCA results to the dataframe
df['cluster'] = clusters
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Step 9: Plot clusters based on PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster', palette='Set2')
plt.title("KMeans Clustering of Songs (PCA Reduced)")
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("cluster_plot.png")  # Saves the plot
plt.close()

# Step 10: Plot playlist genre distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='playlist_genre', order=df['playlist_genre'].value_counts().index, palette='muted')
plt.title("Number of Songs by Playlist Genre")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("genre_distribution.png")  # Saves the plot
plt.close()

# Step 11: Save final dataset with clusters
df.to_csv("spotify_clustered_output.csv", index=False)
