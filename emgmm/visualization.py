import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import numpy as np


def plot_gmm_interactive(X, labels, means, covariances, title="GMM Clustering (Interactive)"):
    fig = go.Figure()

    n_components = means.shape[0]

    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(size=6, color=labels, colorscale='Viridis', opacity=0.7),
        name="Data points"
    ))

    for k in range(n_components):
        mean = means[k]
        cov = covariances[k]

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.arctan2(*vecs[:, 0][::-1])
        t = np.linspace(0, 2 * np.pi, 100)
        ellipse = scale = 2.0 * np.sqrt(vals[:, np.newaxis]) * np.array([np.cos(t), np.sin(t)])
        ellipse = vecs @ ellipse + mean[:, np.newaxis]

        fig.add_trace(go.Scatter(
            x=ellipse[0], y=ellipse[1],
            mode='lines', line=dict(color='black'), name=f"Component {k+1}"
        ))
        fig.add_trace(go.Scatter(
            x=[mean[0]], y=[mean[1]],
            mode='markers', marker=dict(symbol='x', size=10, color='black'),
            showlegend=False
        ))

    fig.update_layout(title=title, width=700, height=600,
                      xaxis_title="X1", yaxis_title="X2", showlegend=True)
    fig.show()


def animate_em_history(X, history, interval=500, save_path=None, max_frames=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_ellipse(mean, cov, ax):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        return Ellipse(xy=mean, width=width, height=height, angle=angle,
                       edgecolor='black', facecolor='none', lw=1.5)

    def update(frame):
        ax.clear()
        ax.set_xlim(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1)
        ax.set_ylim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)

        labels = history["labels"][frame]
        means = history["means"][frame]
        covariances = history["covariances"][frame]

        ax.set_title(f"Iteration {frame + 1}")
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)

        for mean, cov in zip(means, covariances):
            ell = draw_ellipse(mean, cov, ax)
            ax.add_patch(ell)
            ax.plot(mean[0], mean[1], 'kx')

        return ax,

    n_frames = len(history["means"]) if max_frames is None else min(max_frames, len(history["means"]))

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow')
    else:
        plt.close(fig)
        return anim

