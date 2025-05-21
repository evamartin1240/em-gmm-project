import numpy as np

def initialize_parameters(X, n_components, method="kmeans", random_state=42):
    """
    Initialize the parameters of a GMM.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Input data.
    n_components : int
        Number of Gaussians in the mixture.
    method : str, "kmeans" or "random"
        Method for initializing the means.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    weights : ndarray (n_components,)
    means : ndarray (n_components, n_features)
    covariances : ndarray (n_components, n_features, n_features)
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    # Initialize weights uniformly
    weights = np.ones(n_components) / n_components

    # Initialize means
    if method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            # Use KMeans to initialize the means
            kmeans = KMeans(n_clusters=n_components, random_state=random_state)
            kmeans.fit(X)
            means = kmeans.cluster_centers_
        except ImportError:
            # If scikit-learn is not available, use random initialization
            print("scikit-learn not available. Using random initialization.")
            means = X[np.random.choice(n_samples, n_components, replace=False)]
    elif method == "random":
        # Randomly select data points as initial means
        means = X[np.random.choice(n_samples, n_components, replace=False)]
    else:
        raise ValueError(f"Unknown initialization method: {method}")

    # Initialize covariances: identity matrices (plus small regularization)
    covariances = np.array([np.cov(X, rowvar=False) + 1e-6 * np.eye(n_features)
                            for _ in range(n_components)])

    return weights, means, covariances

