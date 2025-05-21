# src/gmm.py
import numpy as np
from scipy.stats import multivariate_normal
from emgmm.utils import initialize_parameters

class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, init_method="kmeans", verbose=False):
        """
        Initialize a Gaussian Mixture Model (GMM) instance.

        Parameters
        ----------
        n_components : int
            Number of Gaussian components in the mixture.
        max_iter : int
            Maximum number of EM iterations.
        tol : float
            Convergence threshold for log-likelihood improvement.
        init_method : str
            Initialization method for means ("kmeans" or "random").
        verbose : bool
            If True, print log-likelihood at each iteration.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.verbose = verbose

        # These will be filled in fit()
        self.weights_ = None        # Mixture weights π_k
        self.means_ = None          # Means μ_k
        self.covariances_ = None    # Covariance matrices Σ_k
        self.log_likelihood_ = []   # Log-likelihood per iteration

        # Store history for visualization or analysis
        self.history_ = {
            "means": [],
            "covariances": [],
            "labels": []
        }

    def fit(self, X):
        """
        Fit the GMM to the data using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        # Parameter initialization
        self._initialize(X)

        log_likelihood_old = None

        for i in range(self.max_iter):
            # E-step: compute responsibilities (gamma)
            gamma = self._e_step(X)

            # M-step: update parameters using current responsibilities
            self._m_step(X, gamma)

            # Store history for this iteration (for visualization/animation)
            labels = np.argmax(gamma, axis=1)
            self.history_["means"].append(self.means_.copy())
            self.history_["covariances"].append(self.covariances_.copy())
            self.history_["labels"].append(labels)

            # Compute log-likelihood for convergence check
            ll = self._compute_log_likelihood(X)
            self.log_likelihood_.append(ll)

            if self.verbose:
                print(f"Iter {i}: log-likelihood = {ll:.4f}")

            # Check for convergence (if log-likelihood improvement is below tolerance)
            if log_likelihood_old is not None and abs(ll - log_likelihood_old) < self.tol:
                break

            log_likelihood_old = ll

        return self

    def predict(self, X):
        """
        Assign each data point to the most probable Gaussian component.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the component each sample belongs to.
        """
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def predict_proba(self, X):
        """
        Return the probability (responsibility) of each point belonging to each component.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        gamma : ndarray of shape (n_samples, n_components)
            Responsibility matrix.
        """
        return self._e_step(X)

    # Internal methods ----------------------

    def _initialize(self, X):
        """
        Initialize weights, means, and covariances using the chosen method.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        """
        self.weights_, self.means_, self.covariances_ = initialize_parameters(X, self.n_components, self.init_method)

    def _e_step(self, X):
        """
        E-step: Compute the responsibilities (posterior probabilities) for each data point.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        gamma : ndarray of shape (n_samples, n_components)
            Responsibility matrix.
        """
        N, D = X.shape
        gamma = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            mvn = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k], allow_singular=True)
            gamma[:, k] = self.weights_[k] * mvn.pdf(X)

        # Normalize responsibilities so that they sum to 1 for each data point
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def _m_step(self, X, gamma):
        """
        M-step: Update the mixture weights, means, and covariances based on responsibilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        gamma : ndarray of shape (n_samples, n_components)
            Responsibility matrix from E-step.
        """
        N, D = X.shape
        Nk = gamma.sum(axis=0)  # Effective number of points assigned to each component

        # Update mixture weights
        self.weights_ = Nk / N
        # Update means
        self.means_ = (gamma.T @ X) / Nk[:, np.newaxis]
        self.covariances_ = []

        # Update covariances for each component
        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = (gamma[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
            cov += 1e-6 * np.eye(D)  # Regularization for numerical stability
            self.covariances_.append(cov)

        self.covariances_ = np.array(self.covariances_)

    def _compute_log_likelihood(self, X):
        """
        Compute the total log-likelihood of the data under the current model parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        log_likelihood : float
            Total log-likelihood of the data.
        """
        N = X.shape[0]
        total = np.zeros(N)

        for k in range(self.n_components):
            mvn = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            total += self.weights_[k] * mvn.pdf(X)

        return np.sum(np.log(total))