import os
import argparse
import numpy as np
import torch
import gymnasium as gym

from experiment_manager import ExperimentManager
from custom_logger import setup_logger, save_json_results, load_json_results
from metrics import (
    permutation_test,
    calculate_convergence_speed,
    calculate_stability,
    compute_moving_average
)
from plotting import (
    plot_learning_curves,
    plot_individual_curves,
    plot_subfigures_compare,
    plot_bar_comparison,
    plot_entropy_evolution,
    plot_failure_analysis
)

# Helper for Random Agent representation
class RandomAgent:
    def __init__(self, action_dim=2, device="cpu"):
        self.action_dim = action_dim
        self.device = torch.device(device)
    def select_action(self, state):
        action = np.random.randint(self.action_dim)
        # Log probability of choosing action uniformly in binary space is log(0.5)
        log_prob = torch.tensor(np.log(1.0 / self.action_dim), dtype=torch.float32, device=self.device)
        entropy = torch.tensor(-np.log(1.0 / self.action_dim), dtype=torch.float32, device=self.device)
        return action, log_prob, entropy
    def update(self, states, rewards, log_probs, entropies):
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "grad_norm": 0.0,
            "mean_entropy": -np.log(1.0 / self.action_dim)
        }

def run_random_agent_experiment(manager, experiment_name, num_seeds, num_episodes):
    """
    Orchestrate training of the random agent for Experiment 10.
    """
    manager.logger.info(f"Starting Random Agent Experiment | Seeds: {num_seeds} | Episodes: {num_episodes}")
    seeds = list(range(42, 42 + num_seeds))
    
    all_rewards = []
    all_lengths = []
    all_policy_losses = []
    all_value_losses = []
    all_grad_norms = []
    all_entropies = []
    
    for seed in seeds:
        manager.set_seeds(seed)
        env = gym.make("CartPole-v1")
        env.action_space.seed(seed)
        
        agent = RandomAgent(action_dim=env.action_space.n)
        
        # Run training loop
        from reinforce_agent import train_agent
        history = train_agent(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            max_steps_per_episode=500,
            logger=None
        )
        env.close()
        
        all_rewards.append(history["rewards"])
        all_lengths.append(history["lengths"])
        all_policy_losses.append(history["policy_losses"])
        all_value_losses.append(history["value_losses"])
        all_grad_norms.append(history["grad_norms"])
        all_entropies.append(history["entropies"])
        
    raw_data = {
        "config": {"agent_type": "random"},
        "seeds": seeds,
        "rewards": all_rewards,
        "lengths": all_lengths,
        "policy_losses": all_policy_losses,
        "value_losses": all_value_losses,
        "grad_norms": all_grad_norms,
        "entropies": all_entropies
    }
    
    results_filepath = os.path.join(manager.results_dir, f"{experiment_name}_raw.json")
    save_json_results(raw_data, results_filepath)
    return raw_data

def main():
    parser = argparse.ArgumentParser(description="REINFORCE Experimental Framework Orchestrator")
    parser.add_argument("--test-mode", action="store_true", help="Run a quick pipeline integration test (few episodes and seeds)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)")
    args = parser.parse_args()

    # Create directories
    results_dir = "./results"
    plots_dir = "./plots"
    logs_dir = "./logs"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Instantiate manager
    manager = ExperimentManager(results_dir=results_dir, logs_dir=logs_dir, device=args.device)
    
    # Configure test vs production scales
    if args.test_mode:
        manager.logger.warning("RUNNING IN TEST MODE: Scaled down seeds and episodes.")
        default_seeds = [42, 43]
        default_episodes = 10
        ex6_lengths = [3, 6, 10]
        num_permutations = 100
    else:
        default_seeds = [42, 43, 44, 45, 46]
        default_episodes = 500
        ex6_lengths = [200, 500, 1000]
        num_permutations = 5000

    manager.logger.info(f"Target Device: {args.device} | Active Seeds: {default_seeds} | Default Episodes: {default_episodes}")

    # =========================================================================
    # EXPERIMENT 1: Gamma Sensitivity
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 1: Gamma Sensitivity ---")
    gamma_values = [0.90, 0.95, 0.99, 0.995]
    ex1_results = {}
    ex1_stats = {}
    
    for gamma in gamma_values:
        config = {
            "lr_policy": 1e-3,
            "gamma": gamma,
            "hidden_sizes": [128, 128],
            "normalize_returns": True,
            "num_episodes": default_episodes
        }
        name = f"gamma_{gamma}"
        raw, stats = manager.run_experiment(f"ex1_{name}", config, seeds=default_seeds)
        ex1_results[f"gamma={gamma}"] = np.array(raw["rewards"])
        ex1_stats[f"gamma={gamma}"] = stats

    # Plots
    plot_learning_curves(
        ex1_results, 
        title="Experiment 1: Gamma Sensitivity - Shaded Mean Reward Curves (95% CI)",
        save_path=os.path.join(plots_dir, "ex1_gamma_comparison.png")
    )
    
    # Plot individual curve for default gamma=0.99
    plot_individual_curves(
        ex1_results["gamma=0.99"],
        title="Experiment 1: Individual Seed Returns for Default Gamma (0.99)",
        save_path=os.path.join(plots_dir, "ex1_individual_default_gamma.png")
    )

    # Bar chart comparisons
    plot_bar_comparison(
        ex1_stats,
        metric_key="mean_final_reward",
        error_key="std_final_reward",
        ylabel="Final Return",
        title="Experiment 1: Final Reward Sensitivity to Discount Factor (Gamma)",
        save_path=os.path.join(plots_dir, "ex1_gamma_bar_rewards.png")
    )
    
    plot_bar_comparison(
        ex1_stats,
        metric_key="mean_convergence_speed",
        error_key=None,
        ylabel="Episode to Reach 475+",
        title="Experiment 1: Convergence Speed Sensitivity to Gamma",
        save_path=os.path.join(plots_dir, "ex1_gamma_bar_convergence.png")
    )

    # =========================================================================
    # EXPERIMENT 2: Learning Rate Sensitivity
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 2: Learning Rate Sensitivity ---")
    lr_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    ex2_results = {}
    ex2_stats = {}
    
    for lr in lr_values:
        config = {
            "lr_policy": lr,
            "gamma": 0.99,
            "hidden_sizes": [128, 128],
            "normalize_returns": True,
            "num_episodes": default_episodes
        }
        name = f"lr_{lr}"
        raw, stats = manager.run_experiment(f"ex2_{name}", config, seeds=default_seeds)
        ex2_results[f"lr={lr}"] = raw
        ex2_stats[f"lr={lr}"] = stats

    # Generate multi-panel reward & loss curves
    plot_subfigures_compare(
        ex2_results,
        metrics_to_plot=[("rewards", "Reward"), ("policy_losses", "Policy Loss")],
        title="Experiment 2: Learning Rate Sensitivity - Reward vs Loss Evolution",
        save_path=os.path.join(plots_dir, "ex2_lr_subplots.png")
    )

    # =========================================================================
    # EXPERIMENT 3: Network Capacity
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 3: Network Capacity ---")
    capacities = [[32, 32], [64, 64], [128, 128], [256, 256]]
    ex3_results = {}
    ex3_stats = {}
    
    for cap in capacities:
        config = {
            "lr_policy": 1e-3,
            "gamma": 0.99,
            "hidden_sizes": cap,
            "normalize_returns": True,
            "num_episodes": default_episodes
        }
        name = f"hidden_{cap[0]}x{cap[1]}"
        raw, stats = manager.run_experiment(f"ex3_{name}", config, seeds=default_seeds)
        ex3_results[f"layers={cap}"] = np.array(raw["rewards"])
        ex3_stats[f"layers={cap}"] = stats

    plot_learning_curves(
        ex3_results,
        title="Experiment 3: Policy Network Capacity Comparison (95% CI)",
        save_path=os.path.join(plots_dir, "ex3_capacity_comparison.png")
    )
    
    plot_bar_comparison(
        ex3_stats,
        metric_key="mean_final_reward",
        error_key="std_final_reward",
        ylabel="Final Return",
        title="Experiment 3: Network Capacity Final Performance Comparison",
        save_path=os.path.join(plots_dir, "ex3_capacity_bar_rewards.png")
    )

    # =========================================================================
    # EXPERIMENT 4: Return Normalization Ablation
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 4: Return Normalization Ablation ---")
    normalization_configs = [True, False]
    ex4_results = {}
    ex4_stats = {}
    
    for norm in normalization_configs:
        config = {
            "lr_policy": 1e-3,
            "gamma": 0.99,
            "hidden_sizes": [128, 128],
            "normalize_returns": norm,
            "num_episodes": default_episodes
        }
        name = "normalized" if norm else "unnormalized"
        raw, stats = manager.run_experiment(f"ex4_{name}", config, seeds=default_seeds)
        ex4_results["With Normalization" if norm else "Without Normalization"] = raw
        ex4_stats["With Normalization" if norm else "Without Normalization"] = stats

    # Side-by-side plot comparing rewards and policy loss gradient/stability
    plot_subfigures_compare(
        ex4_results,
        metrics_to_plot=[("rewards", "Reward"), ("grad_norms", "Gradient Norm")],
        title="Experiment 4: Return Normalization vs Raw Return Learning Dynamics",
        save_path=os.path.join(plots_dir, "ex4_normalization_subplots.png")
    )

    # Pairwise Permutation test for Return Normalization
    rewards_norm = np.array(ex4_results["With Normalization"]["rewards"])[:, -1]
    rewards_unnorm = np.array(ex4_results["Without Normalization"]["rewards"])[:, -1]
    obs_diff, p_val = permutation_test(rewards_norm, rewards_unnorm, num_permutations=num_permutations)
    manager.logger.info(f"Return Normalization Statistical Significance: Obs Diff: {obs_diff:.2f}, P-value: {p_val:.4f}")
    
    save_json_results(
        {"observed_difference": obs_diff, "p_value": p_val, "significant": p_val < 0.05},
        os.path.join(results_dir, "ex4_significance.json")
    )

    # =========================================================================
    # EXPERIMENT 5: Random Seed Study
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 5: Random Seed Study ---")
    # Run on 10 seeds
    seed_study_seeds = default_seeds + ([47, 48, 49, 50, 51] if not args.test_mode else [])
    config_default = {
        "lr_policy": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [128, 128],
        "normalize_returns": True,
        "num_episodes": default_episodes
    }
    raw_ex5, stats_ex5 = manager.run_experiment("ex5_seed_study", config_default, seeds=seed_study_seeds)
    rewards_ex5 = np.array(raw_ex5["rewards"])
    
    # Plot Mean ± Std curve
    plot_learning_curves(
        {"Default Config (10 Seeds)": rewards_ex5},
        title="Experiment 5: Random Seed Study - Mean Return ± 95% Confidence Interval",
        save_path=os.path.join(plots_dir, "ex5_seed_study.png"),
        use_ci=True
    )

    # =========================================================================
    # EXPERIMENT 6: Training Length Comparison
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 6: Training Length Comparison ---")
    # Let's run a single configuration for the longest duration (1000 episodes or max in ex6_lengths)
    max_episodes = max(ex6_lengths)
    config_ex6 = {
        "lr_policy": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [128, 128],
        "normalize_returns": True,
        "num_episodes": max_episodes
    }
    raw_ex6, stats_ex6 = manager.run_experiment("ex6_training_length", config_ex6, seeds=default_seeds)
    rewards_ex6 = np.array(raw_ex6["rewards"])
    
    ex6_analysis = {}
    for length in ex6_lengths:
        subset_rewards = rewards_ex6[:, :length]
        final_returns = subset_rewards[:, -1]
        
        # Calculate sample efficiency (AUC)
        aucs = [float(np.trapz(subset_rewards[s])) for s in range(subset_rewards.shape[0])]
        
        # Calculate convergence speed
        convs = [calculate_convergence_speed(subset_rewards[s], threshold=475.0) for s in range(subset_rewards.shape[0])]
        
        ex6_analysis[f"{length}_episodes"] = {
            "mean_final_reward": float(np.mean(final_returns)),
            "std_final_reward": float(np.std(final_returns, ddof=1) if len(final_returns)>1 else 0.0),
            "mean_auc": float(np.mean(aucs)),
            "mean_convergence": float(np.mean(convs))
        }
        
    save_json_results(ex6_analysis, os.path.join(results_dir, "ex6_length_analysis.json"))
    manager.logger.info(f"Experiment 6 Training Length Analysis Summary: {ex6_analysis}")

    # =========================================================================
    # EXPERIMENT 7: Entropy Analysis
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 7: Entropy Analysis ---")
    # Extract policy entropy from the default config run (from ex5)
    entropies_ex5 = np.array(raw_ex5["entropies"])
    
    plot_entropy_evolution(
        {"Default Config Policy": entropies_ex5},
        save_path=os.path.join(plots_dir, "ex7_entropy_vs_episode.png")
    )
    
    # Calculate initial entropy vs final entropy
    init_entropy = np.mean(entropies_ex5[:, :10])
    final_entropy = np.mean(entropies_ex5[:, -10:])
    manager.logger.info(f"Experiment 7: Initial Entropy: {init_entropy:.4f} | Final Entropy: {final_entropy:.4f}")
    save_json_results(
        {"initial_mean_entropy": init_entropy, "final_mean_entropy": final_entropy},
        os.path.join(results_dir, "ex7_entropy_summary.json")
    )

    # =========================================================================
    # EXPERIMENT 8: Failure Case Analysis
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 8: Failure Case Analysis ---")
    config_failure = {
        "lr_policy": 0.05,
        "gamma": 0.5,
        "hidden_sizes": [128, 128],
        "normalize_returns": True,
        "num_episodes": default_episodes
    }
    raw_fail, stats_fail = manager.run_experiment("ex8_failure_case", config_failure, seeds=default_seeds)
    
    failure_comparison = {
        "Optimal Policy (gamma=0.99, lr=1e-3)": raw_ex5,
        "Failure Policy (gamma=0.50, lr=0.05)": raw_fail
    }
    
    plot_failure_analysis(
        failure_comparison,
        save_path=os.path.join(plots_dir, "ex8_failure_case_analysis.png")
    )

    # =========================================================================
    # EXPERIMENT 9: Hyperparameter Grid Search
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 9: Hyperparameter Grid Search ---")
    
    # Define narrow grid search spaces
    if args.test_mode:
        grid_gammas = [0.95, 0.99]
        grid_lrs = [1e-3, 5e-3]
        grid_hiddens = [[64, 64], [128, 128]]
    else:
        grid_gammas = [0.95, 0.99]
        grid_lrs = [5e-3, 1e-3, 5e-4]
        grid_hiddens = [[64, 64], [128, 128]]
        
    best_score = -float('inf')
    best_config = None
    grid_results = []
    
    for gamma in grid_gammas:
        for lr in grid_lrs:
            for hidden in grid_hiddens:
                config_search = {
                    "lr_policy": lr,
                    "gamma": gamma,
                    "hidden_sizes": hidden,
                    "normalize_returns": True,
                    "num_episodes": default_episodes
                }
                search_name = f"grid_g{gamma}_lr{lr}_h{hidden[0]}x{hidden[1]}"
                # Run with 2 seeds to speed up grid search
                raw, stats = manager.run_experiment(f"ex9_{search_name}", config_search, seeds=default_seeds[:2])
                
                score = stats["mean_final_reward"]
                grid_results.append({
                    "config": config_search,
                    "score": score,
                    "convergence_speed": stats["mean_convergence_speed"],
                    "stability_cv": stats["mean_stability_cv"]
                })
                
                if score > best_score:
                    best_score = score
                    best_config = config_search
                    
    manager.logger.info(f"Grid Search Best Score: {best_score} | Best Config: {best_config}")
    save_json_results(
        {"best_config": best_config, "best_score": best_score, "grid_results": grid_results},
        os.path.join(results_dir, "ex9_grid_search.json")
    )

    # =========================================================================
    # EXPERIMENT 10: REINFORCE vs Random Agent
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 10: REINFORCE vs Random Agent ---")
    raw_random = run_random_agent_experiment(
        manager, 
        "ex10_random_agent", 
        num_seeds=len(default_seeds), 
        num_episodes=default_episodes
    )
    
    rewards_reinforce = np.array(raw_ex5["rewards"])
    rewards_random = np.array(raw_random["rewards"])
    
    # Plot reward comparison
    plot_learning_curves(
        {
            "REINFORCE Agent (Default Config)": rewards_reinforce,
            "Random Uniform Policy Agent": rewards_random
        },
        title="Experiment 10: REINFORCE vs Random Agent Performance (95% CI)",
        save_path=os.path.join(plots_dir, "ex10_reinforce_vs_random.png")
    )

    # =========================================================================
    # EXPERIMENT 11: Reward Scaling and Clipping Sensitivity (Proposed)
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 11: Reward Scaling & Clipping Sensitivity ---")
    scaling_factors = [1.0, 0.1, 0.01]
    ex11_results = {}
    ex11_stats = {}
    
    for scale in scaling_factors:
        config = {
            "lr_policy": 5e-3,  # Run high LR
            "gamma": 0.99,
            "hidden_sizes": [128, 128],
            "normalize_returns": True,
            "reward_scale": scale,
            "num_episodes": default_episodes
        }
        name = f"scale_{scale}"
        raw, stats = manager.run_experiment(f"ex11_{name}", config, seeds=default_seeds)
        ex11_results[f"scale={scale}"] = raw
        ex11_stats[f"scale={scale}"] = stats

    plot_subfigures_compare(
        ex11_results,
        metrics_to_plot=[("rewards", "Reward"), ("grad_norms", "Gradient Norm")],
        title="Experiment 11: Reward Scaling Sensitivity (at Learning Rate = 5e-3)",
        save_path=os.path.join(plots_dir, "ex11_reward_scaling_subplots.png")
    )

    # =========================================================================
    # EXPERIMENT 12: Baseline Variance Reduction Study (Proposed)
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 12: Baseline Variance Reduction Study ---")
    baselines = [None, "moving_average", "value_network"]
    ex12_results = {}
    ex12_stats = {}
    
    for baseline in baselines:
        config = {
            "lr_policy": 1e-3,
            "lr_value": 1e-3,
            "gamma": 0.99,
            "hidden_sizes": [128, 128],
            "normalize_returns": False,  # Turn off return normalization to isolate baseline effect
            "baseline_type": baseline,
            "num_episodes": default_episodes
        }
        name = f"baseline_{baseline if baseline else 'none'}"
        raw, stats = manager.run_experiment(f"ex12_{name}", config, seeds=default_seeds)
        
        label = "No Baseline (Raw Returns)" if baseline is None else f"Baseline: {baseline.replace('_', ' ').title()}"
        ex12_results[label] = raw
        ex12_stats[label] = stats

    # Subfigures comparing reward learning and gradient variance over time
    plot_subfigures_compare(
        ex12_results,
        metrics_to_plot=[("rewards", "Reward"), ("grad_norms", "Gradient Variance (Grad Norm)")],
        title="Experiment 12: Variance Reduction baseline Ablation study",
        save_path=os.path.join(plots_dir, "ex12_baseline_subplots.png")
    )

    # Pairwise statistical permutation test comparing Value Network vs No Baseline
    rewards_val_net = np.array(ex12_results["Baseline: Value Network"]["rewards"])[:, -1]
    rewards_no_base = np.array(ex12_results["No Baseline (Raw Returns)"]["rewards"])[:, -1]
    obs_diff_bl, p_val_bl = permutation_test(rewards_val_net, rewards_no_base, num_permutations=num_permutations)
    manager.logger.info(f"Baseline Significance (Value Net vs Raw): Diff: {obs_diff_bl:.2f}, P-val: {p_val_bl:.4f}")
    save_json_results(
        {"observed_difference": obs_diff_bl, "p_value": p_val_bl, "significant": p_val_bl < 0.05},
        os.path.join(results_dir, "ex12_significance.json")
    )

    # =========================================================================
    # EXPERIMENT 13: Action Space Entropy Regularization Ablation (Proposed)
    # =========================================================================
    manager.logger.info("--- STARTING EXPERIMENT 13: Action Space Entropy Regularization Ablation ---")
    entropy_coefs = [0.0, 0.001, 0.01, 0.1]
    ex13_results = {}
    ex13_stats = {}
    
    for coef in entropy_coefs:
        config = {
            "lr_policy": 1e-3,
            "gamma": 0.99,
            "hidden_sizes": [128, 128],
            "normalize_returns": True,
            "entropy_coef": coef,
            "num_episodes": default_episodes
        }
        name = f"entropy_coef_{coef}"
        raw, stats = manager.run_experiment(f"ex13_{name}", config, seeds=default_seeds)
        ex13_results[f"entropy_coef={coef}"] = raw
        ex13_stats[f"entropy_coef={coef}"] = stats

    plot_subfigures_compare(
        ex13_results,
        metrics_to_plot=[("rewards", "Reward"), ("entropies", "Policy Entropy")],
        title="Experiment 13: Action Space Entropy Regularization Ablation",
        save_path=os.path.join(plots_dir, "ex13_entropy_regularization_subplots.png")
    )

    manager.logger.info("=======================================================")
    manager.logger.info("ALL EXPERIMENTS EXECUTED SUCCESSFULY. FIGURES AND DATA EXPORTED.")
    manager.logger.info("=======================================================")

if __name__ == "__main__":
    main()
