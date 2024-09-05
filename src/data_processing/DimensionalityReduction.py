import os
from sklearn.decomposition import PCA
import numpy as np
import pickle
from ..utils import log_message

class PCAProcessor:
    def __init__(self, n_components=128, model_path=None):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False  # Track whether PCA has been fitted
        self.model_path = model_path  # Path to save/load the PCA model

        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)  # Load model if it exists

    def fit_transform(self, features):
        if not self.is_fitted:
            reduced_features = self.pca.fit_transform(features)
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            log_message('info', f"Explained variance by {self.pca.n_components_} components: {explained_variance:.2f}")
            self.is_fitted = True  # Mark as fitted after fitting
            if self.model_path:
                self.save_model(self.model_path)  # Save model after fitting
            return reduced_features
        else:
            log_message('info', "PCA is already fitted. Using transform instead.")
            return self.transform(features)

    def transform(self, features):
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted yet. Call fit_transform first.")
        return self.pca.transform(features)

    def save_model(self, path):
        """Save the PCA model to the specified path using pickle."""
        # Extract the directory path from the file path
        directory = os.path.dirname(path)

        # Check if the directory exists; if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(path, 'wb') as file:
            pickle.dump(self.pca, file)
        print(f"PCA model saved to {path}")

    def load_model(self, path):
        """Load the PCA model from the specified path using pickle."""
        with open(path, 'rb') as file:
            self.pca = pickle.load(file)
        self.is_fitted = True
        print(f"PCA model loaded from {path}")
