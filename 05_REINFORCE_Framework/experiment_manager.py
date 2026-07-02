import os
import random
import numpy as np
import torch
import gymnasium as gym
from concurrent.futures import ProcessPoolExecutor, as_completed

from reinforce_agent import REINFORCEAgent, train_agent
from custom_logger import setup_logger, save_json_results, load_json_results
from metrics import summarize_experiment_statistics

def seed_all(seed):
    """
    Standard random seeding function.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_seed_helper(seed, config, device):
    """
    Top-level helper function to run a single seed.
    Must be a top-level function to be picklable for multiprocessing.
    """
    seed_all(seed)

    env_name = config.get("env_name", "CartPole-v1")
    env = gym.make(env_name)
    env.action_space.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_policy=config.get("lr_policy", 1e-3),
        lr_value=config.get("lr_value", 1e-3),
        gamma=config.get("gamma", 0.99),
        hidden_sizes=config.get("hidden_sizes", [128, 128]),
        normalize_returns=config.get("normalize_returns", True),
        baseline_type=config.get("baseline_type", None),
        entropy_coef=config.get("entropy_coef", 0.0),
        reward_scale=config.get("reward_scale", 1.0),
        clip_grad_norm=config.get("clip_grad_norm", 10.0),
        device=device
    )
    
    history = train_agent(
        env=env,
        agent=agent,
        num_episodes=config.get("num_episodes", 500),
        max_steps_per_episode=config.get("max_steps_per_episode", 500),
        logger=None
    )
    
    env.close()
    return history

class ExperimentManager:
    """
    Manager to define, execute, log, and analyze RL experiments.
    Supports running seeds in parallel using multiprocessing.
    """
    def __init__(self, results_dir="./results", logs_dir="./logs", device="cpu"):
        self.results_dir = results_dir
        self.logs_dir = logs_dir
        self.device = device
        
        self.logger = setup_logger(
            name="ExperimentManager",
            log_file=os.path.join(logs_dir, "manager.log")
        )
        self.logger.info("Experiment Manager Initialized with Multiprocessing support.")

    def set_seeds(self, seed):
        """
        Public method to seed everything from the manager (used for sequential/custom agents).
        """
        seed_all(seed)

    def run_experiment(self, experiment_name, config, seeds=[42, 43, 44, 45, 46]):
        """
        Run training across multiple seeds in parallel and aggregate results.
        """
        self.logger.info(f"Starting Experiment: {experiment_name} | Runs: {len(seeds)} seeds | Config: {config}")
        
        results_filepath = os.path.join(self.results_dir, f"{experiment_name}_raw.json")
        summary_filepath = os.path.join(self.results_dir, f"{experiment_name}_summary.json")
        
        all_rewards = [None] * len(seeds)
        all_lengths = [None] * len(seeds)
        all_policy_losses = [None] * len(seeds)
        all_value_losses = [None] * len(seeds)
        all_grad_norms = [None] * len(seeds)
        all_entropies = [None] * len(seeds)
        
        # Parallel execution of seeds
        max_workers = min(len(seeds), os.cpu_count() or 4)
        self.logger.info(f"[{experiment_name}] Dispatching {len(seeds)} seeds to {max_workers} processes...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_seed_helper, seed, config, self.device): (idx, seed)
                for idx, seed in enumerate(seeds)
            }
            
            for future in as_completed(futures):
                idx, seed = futures[future]
                try:
                    history = future.result()
                    all_rewards[idx] = history["rewards"]
                    all_lengths[idx] = history["lengths"]
                    all_policy_losses[idx] = history["policy_losses"]
                    all_value_losses[idx] = history["value_losses"]
                    all_grad_norms[idx] = history["grad_norms"]
                    all_entropies[idx] = history["entropies"]
                    
                    final_10_mean = np.mean(history["rewards"][-10:])
                    self.logger.info(f"[{experiment_name}] Seed {seed} Finished. Last 10-Ep Mean Return: {final_10_mean:.2f}")
                except Exception as e:
                    self.logger.error(f"[{experiment_name}] Seed {seed} failed with exception: {e}")
                    raise e

        # Convert to numpy arrays
        rewards_matrix = np.array(all_rewards)
        loss_matrix = np.array(all_policy_losses)
        entropy_matrix = np.array(all_entropies)
        
        # Calculate summary statistics
        stats = summarize_experiment_statistics(rewards_matrix, loss_matrix, entropy_matrix)
        self.logger.info(f"[{experiment_name}] Experiment Complete. Stats Summary: {stats}")
        
        # Bundle data
        raw_data = {
            "config": config,
            "seeds": seeds,
            "rewards": all_rewards,
            "lengths": all_lengths,
            "policy_losses": all_policy_losses,
            "value_losses": all_value_losses,
            "grad_norms": all_grad_norms,
            "entropies": all_entropies
        }
        
        # Save results
        save_json_results(raw_data, results_filepath)
        save_json_results(stats, summary_filepath)
        
        return raw_data, stats
