import os
import numpy as np
import matplotlib.pyplot as plt
from metrics import compute_moving_average, compute_confidence_intervals

# Set professional plotting style parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.autolayout': True
})

# Harmonious color palette
COLORS = [
    '#1F77B4',  # Deep Blue
    '#FF7F0E',  # Orange
    '#2CA02C',  # Forest Green
    '#D62728',  # Crimson Red
    '#9467BD',  # Royal Purple
    '#8C564B',  # Brown
    '#E377C2',  # Muted Pink
    '#7F7F7F',  # Grey
    '#BCBD22',  # Olive
    '#17BECF'   # Teal
]

def plot_learning_curves(
    results_dict,
    metric_name="rewards",
    ylabel="Reward",
    title="Learning Curves",
    save_path=None,
    show=False,
    window=50,
    use_ci=True
):
    """
    Plot learning curves across configurations.
    results_dict: dict of config_name -> numpy array of shape (num_seeds, num_episodes)
    """
    plt.figure(figsize=(10, 6))
    
    for i, (config_name, data_matrix) in enumerate(results_dict.items()):
        color = COLORS[i % len(COLORS)]
        
        # Apply moving average to each seed individually
        smoothed_matrix = np.zeros_like(data_matrix)
        for seed_idx in range(data_matrix.shape[0]):
            smoothed_matrix[seed_idx] = compute_moving_average(data_matrix[seed_idx], window=window)
            
        # Compute mean, standard dev, and confidence intervals
        mean, ci_lower, ci_upper, std = compute_confidence_intervals(smoothed_matrix, confidence=0.95)
        
        x = np.arange(1, len(mean) + 1)
        
        # Plot mean curve
        plt.plot(x, mean, label=config_name, color=color, linewidth=2.0)
        
        # Plot error band
        if use_ci:
            plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
        else:
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            
    plt.xlabel("Episode")
    plt.ylabel(f"{ylabel} (Moving Avg, window={window})")
    plt.title(title)
    plt.legend(loc="best", frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_individual_curves(
    data_matrix,
    title="Individual Run Curves",
    save_path=None,
    show=False,
    window=50
):
    """
    Plot individual seed curves as light lines and their mean as a bold line.
    data_matrix: numpy array of shape (num_seeds, num_episodes)
    """
    plt.figure(figsize=(10, 6))
    num_seeds = data_matrix.shape[0]
    color = COLORS[0]
    
    x = np.arange(1, data_matrix.shape[1] + 1)
    
    # Plot individual runs
    for s in range(num_seeds):
        smoothed_run = compute_moving_average(data_matrix[s], window=window)
        plt.plot(x, smoothed_run, color=color, alpha=0.3, linewidth=1.0, label=f"Seed {s+1}" if s == 0 else "")
        
    # Plot mean run
    mean = np.mean(data_matrix, axis=0)
    smoothed_mean = compute_moving_average(mean, window=window)
    plt.plot(x, smoothed_mean, color=color, linewidth=2.5, label="Mean Return")
    
    plt.xlabel("Episode")
    plt.ylabel(f"Reward (Moving Avg, window={window})")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_subfigures_compare(
    results_dict,
    metrics_to_plot=[("rewards", "Reward"), ("policy_losses", "Policy Loss")],
    title="Multi-Metric Comparison",
    save_path=None,
    show=False,
    window=50
):
    """
    Create a multi-panel plot comparing multiple metrics across configurations.
    results_dict: dict of config_name -> dict of metric_name -> numpy array (num_seeds, num_episodes)
    metrics_to_plot: list of tuples (metric_key, metric_label)
    """
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
        
    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        
        for i, (config_name, metrics) in enumerate(results_dict.items()):
            if metric_key not in metrics:
                continue
            data_matrix = np.array(metrics[metric_key])
            color = COLORS[i % len(COLORS)]
            
            # Smooth
            smoothed_matrix = np.zeros_like(data_matrix)
            for s in range(data_matrix.shape[0]):
                smoothed_matrix[s] = compute_moving_average(data_matrix[s], window=window)
                
            mean, ci_lower, ci_upper, _ = compute_confidence_intervals(smoothed_matrix)
            x = np.arange(1, len(mean) + 1)
            
            ax.plot(x, mean, label=config_name, color=color, linewidth=2.0)
            ax.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.15)
            
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"{metric_label} (Smoothed)")
        ax.set_title(f"Comparison: {metric_label}")
        ax.grid(True)
        if ax_idx == 0:
            ax.legend(loc="best", frameon=True)
            
    fig.suptitle(title, y=1.02)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_bar_comparison(
    stats_dict,
    metric_key="mean_final_reward",
    error_key="std_final_reward",
    ylabel="Reward",
    title="Performance Comparison",
    save_path=None,
    show=False
):
    """
    Plot a bar chart comparing a metric value across configs, with error bars.
    stats_dict: dict of config_name -> dict of computed statistics
    """
    plt.figure(figsize=(10, 5))
    
    configs = list(stats_dict.keys())
    values = [stats_dict[c][metric_key] for c in configs]
    errors = [stats_dict[c][error_key] if error_key in stats_dict[c] else 0.0 for c in configs]
    
    bars = plt.bar(configs, values, yerr=errors, color=COLORS[:len(configs)], capsize=8, alpha=0.85, edgecolor='black', linewidth=1.0)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0,
            yval + (max(values)*0.01),
            f"{yval:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            weight='bold'
        )
        
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_entropy_evolution(
    entropy_results,
    save_path=None,
    show=False,
    window=50
):
    """
    Plot policy entropy curves vs episode.
    """
    plot_learning_curves(
        entropy_results,
        metric_name="entropies",
        ylabel="Entropy (Nats)",
        title="Policy Entropy Evolution vs Episode",
        save_path=save_path,
        show=show,
        window=window
    )

def plot_failure_analysis(
    results_dict,
    save_path=None,
    show=False,
    window=50
):
    """
    Generate subplots specifically analyzing the failure case.
    """
    # Expect keys like "Optimal (gamma=0.99, lr=1e-3)", "Failure (gamma=0.50, lr=0.05)"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        ("rewards", "Rewards", "Reward (Moving Avg)"),
        ("policy_losses", "Policy Loss", "Loss (Smoothed)"),
        ("grad_norms", "Gradient Norms", "Grad Norm (Smoothed)")
    ]
    
    for idx, (key, label, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        for i, (config_name, data) in enumerate(results_dict.items()):
            if key not in data:
                continue
            data_matrix = np.array(data[key])
            color = COLORS[i % len(COLORS)]
            
            smoothed_matrix = np.zeros_like(data_matrix)
            for s in range(data_matrix.shape[0]):
                smoothed_matrix[s] = compute_moving_average(data_matrix[s], window=window)
                
            mean = np.mean(smoothed_matrix, axis=0)
            x = np.arange(1, len(mean) + 1)
            
            ax.plot(x, mean, label=config_name, color=color, linewidth=2.0)
            
            # Check if standard deviation exists and plot
            if data_matrix.shape[0] > 1:
                std = np.std(smoothed_matrix, axis=0)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
                
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(True)
        if idx == 0:
            ax.legend(loc="best")
            
    fig.suptitle("Failure Case Mechanisms Analysis", y=1.03)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close()
