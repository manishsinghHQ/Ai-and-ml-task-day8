import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load and visualize dataset and PCA for 2D view
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

#  PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])

# 2.Fit K-Means and assign cluster labels 
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
X_pca_df['Cluster'] = clusters

# 3. Use the Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# 4.Visualize clusters with color-coding using PCA 2D projection
plt.figure(figsize=(6, 4))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=X_pca_df, palette='Set1')
plt.title('Cluster Visualization with K-Means')
plt.show()

# 5.Evaluate clustering using Silhouette Score
score = silhouette_score(X, clusters)
print(f'Silhouette Score: {score:.2f}')
