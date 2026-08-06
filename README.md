# Spotify Song Segmentation 🎧

This project focuses on analyzing Spotify songs and grouping them into similar clusters based on their audio characteristics using **Unsupervised Machine Learning** techniques.

The project uses **KMeans Clustering** to identify hidden patterns in songs and **PCA (Principal Component Analysis)** for visualization.

---

## 📌 Project Files

| File | Description |
|---|---|
| spotify_genre_segmentation.py | Main Python script containing preprocessing, clustering and visualization |
| spotify dataset.csv | Original Spotify dataset |
| spotify_clustered_output.csv | Final dataset with cluster labels assigned to songs |
| cluster_summary.csv | Average audio feature values for each cluster |
| cluster_genre_mapping.csv | Dominant genre mapping for each cluster |
| correlation_matrix.png | Heatmap showing correlation between audio features |
| cluster_plot.png | PCA-based visualization of song clusters |
| genre_distribution.png | Distribution of songs across genres |

---

## 🎵 Features Used for Clustering

The following audio features were used:

- Danceability
- Energy
- Key
- Loudness
- Mode
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration

---

## ⚙️ Techniques Used

- Data Cleaning & Preprocessing
- Feature Scaling using StandardScaler
- Unsupervised Learning
  - KMeans Clustering
  - PCA for Dimensionality Reduction
- Data Visualization using Matplotlib and Seaborn

---

## 🔍 Project Workflow

1. Loaded Spotify dataset
2. Removed missing song information
3. Selected important audio features
4. Standardized features
5. Applied KMeans clustering
6. Reduced dimensions using PCA for visualization
7. Analyzed clusters using feature averages and genre mapping
8. Generated final clustered dataset

---

## 📊 Output

The model groups songs into **6 clusters** based on similarity in their audio features.

These clusters do not have predefined labels. Each cluster represents a group of songs with similar characteristics such as energy, danceability, tempo and acousticness.

---

## 🛠️ Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 👨‍💻 Author

Sumit Chauhan
