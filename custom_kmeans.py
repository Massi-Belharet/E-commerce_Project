import numpy as np
from typing import Tuple

class CustomKMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        """
        Initialize KMeans instance
        
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids by randomly selecting points from the dataset
        
        """
        n_samples = X.shape[0]
        rng = np.random.RandomState(42)  # For reproducibility
        indices = rng.permutation(n_samples)[:self.n_clusters]
        return X[indices]
    
    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid using vectorized operations
        
        """
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroid positions using vectorized operations
        
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = X[mask].mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[k] = X[np.random.randint(X.shape[0])]
        return centroids
    
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit KMeans to the data using vectorized operations
        
        """
        X = np.asarray(X)
        
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iter):
            # Store old centroids
            old_centroids = self.centroids.copy()
            
            # Assign clusters
            self.labels = self.assign_clusters(X, self.centroids)
            
            # Update centroids
            self.centroids = self.update_centroids(X, self.labels)
            
            if np.allclose(old_centroids, self.centroids):
                break
                
        return self.labels, self.centroids
    