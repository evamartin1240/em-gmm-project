import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
from emgmm.gmm import GMM

# Runs multiple GMM experiments with different initializations and collects metrics
def run_multiple_experiments(X, y_true=None, n_runs=10, n_components=3, max_iter=100, tol=1e-4):
    # Initialize results dictionary to store metrics for each method
    results = {
        "kmeans": {"log_likelihood": [], "iterations": [], "ari": []},
        "random": {"log_likelihood": [], "iterations": [], "ari": []}
    }

    # Loop over different random seeds for reproducibility
    for seed in tqdm(range(n_runs), desc="Running experiments"):
        # Test both initialization methods: kmeans and random
        for method in ["kmeans", "random"]:
            # Create GMM instance with specified initialization
            gmm = GMM(n_components=n_components, max_iter=max_iter, tol=tol,
                      init_method=method, verbose=False)
            np.random.seed(seed)  # Set random seed for reproducibility
            gmm.fit(X)  # Fit GMM to data

            labels = gmm.predict(X)  # Predict cluster labels
            loglik = gmm.log_likelihood_[-1]  # Get final log-likelihood
            iters = len(gmm.log_likelihood_)  # Get number of iterations

            # Store metrics for this run
            results[method]["log_likelihood"].append(loglik)
            results[method]["iterations"].append(iters)

            # If ground truth labels are provided, compute ARI
            if y_true is not None:
                ari = adjusted_rand_score(y_true, labels)
                results[method]["ari"].append(ari)

    return results

# Plots boxplots comparing metrics between initialization methods
def plot_results_comparison(results):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    # Iterate over each metric to plot
    for i, metric in enumerate(["log_likelihood", "iterations", "ari"]):
        if not results["kmeans"][metric]:
            continue  # Skip if no results for this metric

        # Create boxplot for both initialization methods
        axs[i].boxplot([results["kmeans"][metric], results["random"][metric]],
                       labels=["KMeans Init", "Random Init"])
        axs[i].set_title(metric.replace("_", " ").title())
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

