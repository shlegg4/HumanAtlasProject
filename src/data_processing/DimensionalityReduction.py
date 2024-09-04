from sklearn.decomposition import PCA
import numpy as np
from ..utils import log_message

class PCAProcessor:
    def __init__(self, n_components=128):
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, features):
        reduced_features = self.pca.fit_transform(features)
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"Explained variance by {self.pca.n_components_} components: {explained_variance:.2f}")
        return reduced_features

    def transform(self, features):
        return self.pca.transform(features)