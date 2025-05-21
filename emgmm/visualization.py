import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import numpy as np


def plot_gmm_interactive(X, labels, means, covariances, title="GMM Clustering (Interactive)"):
    """
    Plot GMM clustering results interactively using Plotly.

    Parameters
    ----------
    X : ndarray
        Data points.
    labels : ndarray
        Cluster assignments for each data point.
    means : ndarray
        Means of the Gaussian components.
    covariances : ndarray
        Covariance matrices of the Gaussian components.
    title : str
        Title for the plot.
    """
    fig = go.Figure()

    n_components = means.shape[0]

    # Plot data points colored by cluster label
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(size=6, color=labels, colorscale='Viridis', opacity=0.7),
        name="Data points"
    ))

    # Plot ellipses for each Gaussian component
    for k in range(n_components):
        mean = means[k]
        cov = covariances[k]

        # Calculate ellipse for the covariance
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.arctan2(*vecs[:, 0][::-1])
        t = np.linspace(0, 2 * np.pi, 100)
        ellipse = scale = 2.0 * np.sqrt(vals[:, np.newaxis]) * np.array([np.cos(t), np.sin(t)])
        ellipse = vecs @ ellipse + mean[:, np.newaxis]

        # Add ellipse outline
        fig.add_trace(go.Scatter(
            x=ellipse[0], y=ellipse[1],
            mode='lines', line=dict(color='black'), name=f"Component {k+1}"
        ))
        # Add mean marker
        fig.add_trace(go.Scatter(
            x=[mean[0]], y=[mean[1]],
            mode='markers', marker=dict(symbol='x', size=10, color='black'),
            showlegend=False
        ))

    fig.update_layout(title=title, width=700, height=600,
                      xaxis_title="X1", yaxis_title="X2", showlegend=True)
    fig.show()


def animate_em_history(X, history, interval=500, save_path=None, max_frames=None):
    """
    Animate the EM algorithm's progress over iterations using matplotlib.

    Parameters
    ----------
    X : ndarray
        Data points.
    history : dict
        Dictionary containing lists of means, covariances, and labels for each iteration.
    interval : int
        Delay between frames in milliseconds.
    save_path : str or None
        If provided, save the animation to this path.
    max_frames : int or None
        Maximum number of frames to animate.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_ellipse(mean, cov, ax):
        """
        Draw an ellipse representing a Gaussian component.
        """
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        return Ellipse(xy=mean, width=width, height=height, angle=angle,
                       edgecolor='black', facecolor='none', lw=1.5)

    def update(frame):
        """
        Update function for each animation frame.
        """
        ax.clear()
        # Set axis limits based on data
        ax.set_xlim(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1)
        ax.set_ylim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)

        # Get current iteration's labels, means, and covariances
        labels = history["labels"][frame]
        means = history["means"][frame]
        covariances = history["covariances"][frame]

        ax.set_title(f"Iteration {frame + 1}")
        # Plot data points colored by cluster assignment
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)

        # Draw ellipses and means for each component
        for mean, cov in zip(means, covariances):
            ell = draw_ellipse(mean, cov, ax)
            ax.add_patch(ell)
            ax.plot(mean[0], mean[1], 'kx')

        return ax,

    # Determine number of frames to animate
    n_frames = len(history["means"]) if max_frames is None else min(max_frames, len(history["means"]))

    # Create the animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        # Save animation to file if path is provided
        anim.save(save_path, writer='pillow')
    else:
        # Otherwise, return the animation object for inline display
        plt.close(fig)
        return anim

