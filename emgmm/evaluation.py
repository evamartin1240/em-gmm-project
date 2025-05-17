import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
from emgmm.gmm import GMM

def run_multiple_experiments(X, y_true=None, n_runs=10, n_components=3, max_iter=100, tol=1e-4):
    results = {
        "kmeans": {"log_likelihood": [], "iterations": [], "ari": []},
        "random": {"log_likelihood": [], "iterations": [], "ari": []}
    }

    for seed in tqdm(range(n_runs), desc="Running experiments"):
        for method in ["kmeans", "random"]:
            gmm = GMM(n_components=n_components, max_iter=max_iter, tol=tol,
                      init_method=method, verbose=False)
            np.random.seed(seed)
            gmm.fit(X)

            labels = gmm.predict(X)
            loglik = gmm.log_likelihood_[-1]
            iters = len(gmm.log_likelihood_)

            results[method]["log_likelihood"].append(loglik)
            results[method]["iterations"].append(iters)

            if y_true is not None:
                ari = adjusted_rand_score(y_true, labels)
                results[method]["ari"].append(ari)

    return results


def plot_results_comparison(results):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    for i, metric in enumerate(["log_likelihood", "iterations", "ari"]):
        if not results["kmeans"][metric]:
            continue

        axs[i].boxplot([results["kmeans"][metric], results["random"][metric]],
                       labels=["KMeans Init", "Random Init"])
        axs[i].set_title(metric.replace("_", " ").title())
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

