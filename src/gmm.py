# src/gmm.py
import numpy as np
from scipy.stats import multivariate_normal
from src.utils import initialize_parameters

class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, init_method="kmeans", verbose=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.verbose = verbose

        # Se llenarán en fit()
        self.weights_ = None        # π_k (pesos de mezcla)
        self.means_ = None          # μ_k (medias)
        self.covariances_ = None    # Σ_k (matrices de covarianza)
        self.log_likelihood_ = []   # Log-likelihood por iteración

    def fit(self, X):
        """
        Entrena el modelo GMM usando el algoritmo EM.
        """
        # Inicialización de parámetros
        self._initialize(X)

        log_likelihood_old = None

        for i in range(self.max_iter):
            # E-step: calcular responsabilidades
            gamma = self._e_step(X)

            # M-step: actualizar parámetros
            self._m_step(X, gamma)

            # Log-verosimilitud
            ll = self._compute_log_likelihood(X)
            self.log_likelihood_.append(ll)

            if self.verbose:
                print(f"Iter {i}: log-likelihood = {ll:.4f}")

            # Criterio de convergencia
            if log_likelihood_old is not None and abs(ll - log_likelihood_old) < self.tol:
                break

            log_likelihood_old = ll

        return self

    def predict(self, X):
        """
        Asigna a cada punto el componente más probable.
        """
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def predict_proba(self, X):
        """
        Devuelve las probabilidades de pertenencia a cada componente.
        """
        return self._e_step(X)

    # Métodos internos ----------------------

    def _initialize(self, X):
        self.weights_, self.means_, self.covariances_ = initialize_parameters(X, self.n_components, self.init_method)

    def _e_step(self, X):
        N, D = X.shape
        gamma = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            mvn = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            gamma[:, k] = self.weights_[k] * mvn.pdf(X)

        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def _m_step(self, X, gamma):
        N, D = X.shape
        Nk = gamma.sum(axis=0)

        self.weights_ = Nk / N
        self.means_ = (gamma.T @ X) / Nk[:, np.newaxis]
        self.covariances_ = []

        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = (gamma[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
            self.covariances_.append(cov)

        self.covariances_ = np.array(self.covariances_)

    def _compute_log_likelihood(self, X):
        N = X.shape[0]
        total = np.zeros(N)

        for k in range(self.n_components):
            mvn = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            total += self.weights_[k] * mvn.pdf(X)

        return np.sum(np.log(total))

