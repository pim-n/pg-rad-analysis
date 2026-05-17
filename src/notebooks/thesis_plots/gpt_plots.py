import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def count_rate_alignment(color='red'):
    A, n_gamma, mu_air, epsilon = 100e6, 1.0, 0.0001, 1.0
    y0 = 32.0
    x_min, x_max = -100, 100
    bin_width, dx_fine = 15.0, 0.05

    def count_rate(x):
        r = np.sqrt(x**2 + y0**2)
        return epsilon * A * n_gamma * np.exp(-mu_air * r) / (4 * np.pi * r**2)

    def compute_bins(edges):
        centers = 0.5 * (edges[:-1] + edges[1:])
        avg_rates, cumulative_counts = [], []
        for x0, x1 in zip(edges[:-1], edges[1:]):
            xs = np.linspace(x0, x1, 2000)
            rates = count_rate(xs)
            avg_rates.append(np.mean(rates))
            cumulative_counts.append(np.trapezoid(rates, xs) / bin_width)  # Normalize by bin width
        return centers, np.array(avg_rates), np.array(cumulative_counts)

    best_edges = np.arange(-7.5 - 20 * bin_width, 7.5 + 20 * bin_width, bin_width)
    worst_edges = np.arange(-20 * bin_width, 20 * bin_width + bin_width, bin_width)
    best_centers, best_rates, best_counts = compute_bins(best_edges)
    worst_centers, worst_rates, worst_counts = compute_bins(worst_edges)
    x_fine = np.arange(x_min, x_max + dx_fine, dx_fine)
    y_fine = count_rate(x_fine)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    cases = [
        (axes[0], "Best alignment", best_edges, best_centers, best_rates, best_counts),
        (axes[1], "Worst alignment", worst_edges, worst_centers, worst_rates, worst_counts),
    ]

    for ax, title, edges, centers, rates, counts in cases:
        for x0, x1 in zip(edges[:-1], edges[1:]):
            ax.add_patch(Rectangle((x0, 0), x1 - x0, 20, facecolor="lightgray", edgecolor="black", linewidth=1, alpha=0.25))
        ax.plot(x_fine, y_fine, color=color, linewidth=2.8, zorder=5)
        ax.step(edges[:-1], rates, where="post", color="black", linewidth=2.5, zorder=6)
        for xc, yc, c in zip(centers, rates, counts):
            if yc > 0.15:
                ax.text(xc, yc + 0.25, f"{c:.1f}", fontsize=9, rotation=90, ha="center", va="bottom")
        ax.set_title(title, fontsize=24)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 20)
        ax.set_xlabel("x coordinate (m)", fontsize=24)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Count-rate (1/s)", fontsize=26)
    plt.tight_layout()

def trajectory_example(color='red'):
    fig, ax = plt.subplots(figsize=(7, 7))
    x_line = np.linspace(0, 10, 400)
    y_line = -x_line + 7
    ax.plot(x_line, y_line, color="black", linewidth=1.5)
    source = np.array([5.0, 5.0])
    ax.scatter(source[0], source[1], s=120, color="black", zorder=5)
    detector = np.array([1.8, 5.0])
    ax.scatter(detector[0], detector[1], s=120, facecolors="none", edgecolors="black", linewidths=1.5, zorder=6)
    ax.plot([detector[0], source[0]], [detector[1], source[1]], color=color, linewidth=2.5)
    ax.text(3.6, 5.35, r"$r$", fontsize=28)
    x0, y0 = source
    a, b, c = 1, 1, -7
    denom = a**2 + b**2
    x_proj = x0 - a * (a*x0 + b*y0 + c) / denom
    y_proj = y0 - b * (a*x0 + b*y0 + c) / denom
    ax.plot([source[0], x_proj], [source[1], y_proj], linestyle=(0, (4, 3)), color="black", linewidth=1.5)
    mid = 0.5 * (source + np.array([x_proj, y_proj]))
    ax.text(mid[0] - 0.2, mid[1] + 0.2, r"$d$", fontsize=28)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate", fontsize=28)
    ax.set_ylabel("y coordinate", fontsize=28)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.75)
        spine.set_color("0.25")
    plt.tight_layout()
    plt.show()