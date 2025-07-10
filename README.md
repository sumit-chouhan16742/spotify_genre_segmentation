# Spotify Genre Segmentation 🎧

This project is submitted as part of my **Artificial Intelligence Internship** at **Corizo**.

It involves analyzing a large Spotify dataset to segment songs into different genre-like clusters using unsupervised learning techniques such as **KMeans Clustering** and **PCA (Principal Component Analysis)**.

---

## 📁 Project Files

| File                          | Description                                 |
|-------------------------------|---------------------------------------------|
| `spotify_genre_segmentation.py` | Main Python script with preprocessing, clustering, and visualizations |
| `spotify dataset.csv`           | Original dataset provided                   |
| `spotify_clustered_output.csv`  | Final output dataset with cluster labels    |
| `correlation_matrix.png`        | Heatmap showing correlation between audio features |
| `cluster_plot.png`              | 2D PCA-based plot of song clusters          |
| `genre_distribution.png`        | Bar chart showing number of songs by genre  |

---

## 📊 Features Used for Clustering
- `danceability`
- `energy`
- `key`
- `loudness`
- `mode`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_ms`

---

## ⚙️ Techniques Used
- Data Cleaning & Preprocessing
- Feature Scaling (StandardScaler)
- Unsupervised Learning:
  - KMeans Clustering
  - PCA for dimensionality reduction
