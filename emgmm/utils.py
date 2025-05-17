import numpy as np

def initialize_parameters(X, n_components, method="kmeans", random_state=42):
    """
    Inicializa los parámetros de una GMM.
    
    Parámetros
    ----------
    X : ndarray (n_samples, n_features)
        Datos de entrada.
    n_components : int
        Número de gaussianas en la mezcla.
    method : str, "kmeans" o "random"
        Método de inicialización de las medias.
    random_state : int
        Semilla para reproducibilidad.
    
    Retorna
    -------
    weights : ndarray (n_components,)
    means : ndarray (n_components, n_features)
    covariances : ndarray (n_components, n_features, n_features)
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    # Inicializamos pesos uniformemente
    weights = np.ones(n_components) / n_components

    # Inicialización de medias
    if method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_components, random_state=random_state)
            kmeans.fit(X)
            means = kmeans.cluster_centers_
        except ImportError:
            print("scikit-learn no disponible. Usando inicialización aleatoria.")
            means = X[np.random.choice(n_samples, n_components, replace=False)]
    elif method == "random":
        means = X[np.random.choice(n_samples, n_components, replace=False)]
    else:
        raise ValueError(f"Método de inicialización desconocido: {method}")

    # Inicialización de covarianzas: matrices identidad
    covariances = np.array([np.cov(X, rowvar=False) + 1e-6 * np.eye(n_features)
                            for _ in range(n_components)])

    return weights, means, covariances

