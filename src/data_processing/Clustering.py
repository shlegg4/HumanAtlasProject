from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class ClusteringProcessor:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def fit(self, data):
        self.kmeans.fit(data)
        return self.kmeans.labels_
    
    def plot_clusters(self, data, labels):
        plt.figure(figsize=(10, 7))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.title(f'K-Means Clustering with {self.n_clusters} Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()
