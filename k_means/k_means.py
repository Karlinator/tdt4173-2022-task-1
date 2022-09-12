import numpy as np
import numpy.typing as npt
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    iterations: int

    k: int
    centroids: npt.ArrayLike
    
    def __init__(self, iterations=10, k=2):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.iterations = iterations
        self.k = k
        
    def fit(self, X: npt.NDArray) -> None:
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        X = np.array(X) # Enure we get actual ndarray, and not some other stupid type.
        self.centroids = np.random.random_sample(size=(self.k, 2))
        for iteration in range(self.iterations):
            # Put samples in groups
            distances = cross_euclidean_distance(X, self.centroids)
            groups = [[]] * self.k
            for j in range(self.k):
                groups[j] = np.array([X[i] for i, x in enumerate(distances) if min(x) == x[j]])

            for i, group in enumerate(groups):
                if len(group) == 0:
                    # A group is empty
                    # This can cause problems, so we re-initialize that centroid.
                    # If the empty group persists until the end it will cause a crash elsewhere,
                    # and en ampty group is not being useful (we know how many groups there is supposed to be, after all)
                    print(f"Group {i} is empty at iteration {iteration}, re-initializing")
                    self.centroids[i] = np.random.random_sample(size=(1, 2))

            if iteration % 10 == 0:
                print (f"Finished iteration {iteration}")

            # Update centroids
            self.centroids = np.array([[x / len(group) for x in np.sum(group, axis=0)] if len(group) > 0 else self.centroids[i] for i, group in enumerate(groups)])


    
    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        distances = cross_euclidean_distance(np.array(X), self.centroids)
        return np.array([min(range(len(distance)), key=distance.__getitem__) for  distance in distances])
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids

    
def normalize(X: pd.DataFrame) -> pd.DataFrame:
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x: np.ndarray[float, float], y: np.ndarray[float, float] | None = None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))


def main():
    data_2 = pd.read_csv('k_means/data_2.csv')[:10]
    print(normalize(data_2))
    # data_2.describe().T
    # X = data_2[['x0', 'x1']][:10]
    # model_2 = KMeans(k=10, iterations=10)  # <-- Feel free to add hyperparameters 
    # model_2.fit(X)
    # print(model_2.predict(X))
    # print(X)
    # print(model_2.get_centroids())

if __name__ == '__main__':
    main()
