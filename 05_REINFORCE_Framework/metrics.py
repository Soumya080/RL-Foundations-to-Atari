import numpy as np

def compute_moving_average(data, window=50):
    """
    Compute the moving average of a 1D array.
    """
    if len(data) < window:
        # Fallback if trajectory is shorter than window
        return np.array([np.mean(data[:i+1]) for i in range(len(data))])
    
    # Standard convolution moving average
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    # Prepend padding so returned array has same length as original data
    padding = np.array([np.mean(data[:i+1]) for i in range(window - 1)])
    return np.concatenate([padding, ma])

def compute_confidence_intervals(data_matrix, confidence=0.95):
    """
    Compute mean and confidence intervals across seeds.
    Args:
        data_matrix: numpy array of shape (num_seeds, num_episodes)
        confidence: confidence level (default: 0.95)
    Returns:
        mean: shape (num_episodes,)
        lower_bound: shape (num_episodes,)
        upper_bound: shape (num_episodes,)
        std: shape (num_episodes,)
    """
    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0, ddof=1) if data_matrix.shape[0] > 1 else np.zeros_like(mean)
    n = data_matrix.shape[0]
    
    # 95% Confidence Interval z-critical value is 1.96
    # 90% is 1.645, 99% is 2.576
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.90:
        z = 1.645
    elif confidence == 0.99:
        z = 2.576
    else:
        z = 1.96
        
    margin_of_error = z * (std / np.sqrt(n)) if n > 1 else np.zeros_like(mean)
    return mean, mean - margin_of_error, mean + margin_of_error, std

def calculate_convergence_speed(rewards, threshold=475.0, window=100):
    """
    Determine the episode index where the moving average first crosses a threshold.
    Returns:
        episode_index: int (the first episode where threshold is achieved, or length of rewards if never achieved)
    """
    moving_avg = compute_moving_average(rewards, window=window)
    crossings = np.where(moving_avg >= threshold)[0]
    if len(crossings) > 0:
        return int(crossings[0])
    return len(rewards)

def calculate_stability(rewards, late_window=100):
    """
    Measure stability as the standard deviation and coefficient of variation of rewards
    in the last N episodes of training.
    """
    if len(rewards) < late_window:
        late_data = np.array(rewards)
    else:
        late_data = np.array(rewards[-late_window:])
        
    std = np.std(late_data)
    mean = np.mean(late_data)
    
    # Coefficient of variation (CV = std / mean) - normalized measure of dispersion
    cv = std / (mean + 1e-8)
    return {
        "late_mean": mean,
        "late_std": std,
        "coeff_of_variation": cv
    }

def check_overfitting(rewards, window=50):
    """
    Analyze if the agent shows signs of overfitting/performance decay by comparing the
    peak moving average reward to the final moving average reward.
    """
    moving_avg = compute_moving_average(rewards, window=window)
    peak_val = np.max(moving_avg)
    peak_idx = np.argmax(moving_avg)
    final_val = moving_avg[-1]
    
    abs_drop = peak_val - final_val
    pct_drop = (abs_drop / (peak_val + 1e-8)) * 100.0
    
    # Overfitting is defined here as a significant drop (> 15%) from peak performance late in training
    is_overfitting = pct_drop > 15.0 and peak_idx < (len(rewards) * 0.8)
    
    return {
        "peak_value": peak_val,
        "peak_episode": int(peak_idx),
        "final_value": final_val,
        "absolute_drop": abs_drop,
        "percent_drop": pct_drop,
        "is_overfitting": bool(is_overfitting)
    }

def calculate_sample_efficiency(rewards):
    """
    Compute Area Under Curve (AUC) of rewards as a metric for sample efficiency.
    Higher AUC means the agent learned faster (more reward accumulated early).
    """
    return float(np.trapz(rewards))

def calculate_divergence_rate(rewards_matrix, success_threshold=400.0, late_window=50):
    """
    Calculate the percentage of seeds that failed to learn (diverged/failed cases).
    """
    num_seeds = rewards_matrix.shape[0]
    failed_seeds = 0
    for i in range(num_seeds):
        late_mean = np.mean(rewards_matrix[i, -late_window:])
        if late_mean < success_threshold:
            failed_seeds += 1
    return (failed_seeds / num_seeds) * 100.0

def permutation_test(group1, group2, num_permutations=10000, seed=42):
    """
    Non-parametric bootstrap permutation test to compare the means of two groups.
    Does not assume normality. Excellent for small sample sizes typical in RL (5-10 seeds).
    
    Null Hypothesis: The means of both groups are identical.
    Alternative Hypothesis: The means are different (two-tailed test).
    
    Returns:
        observed_diff: Absolute difference between sample means.
        p_value: Probability of observing a difference as extreme under the null hypothesis.
    """
    np.random.seed(seed)
    g1 = np.array(group1)
    g2 = np.array(group2)
    n1 = len(g1)
    n2 = len(g2)
    
    observed_diff = np.abs(np.mean(g1) - np.mean(g2))
    combined = np.concatenate([g1, g2])
    
    count = 0
    for _ in range(num_permutations):
        perm = np.random.permutation(combined)
        perm_g1 = perm[:n1]
        perm_g2 = perm[n1:]
        perm_diff = np.abs(np.mean(perm_g1) - np.mean(perm_g2))
        if perm_diff >= observed_diff:
            count += 1
            
    p_value = count / num_permutations
    return observed_diff, p_value

def summarize_experiment_statistics(rewards_matrix, loss_matrix, entropy_matrix, convergence_threshold=475.0):
    """
    Summarize key statistics across all runs for a single configuration.
    """
    num_seeds = rewards_matrix.shape[0]
    
    final_rewards = rewards_matrix[:, -1]
    mean_final_reward = np.mean(final_rewards)
    std_final_reward = np.std(final_rewards, ddof=1) if num_seeds > 1 else 0.0
    
    convergence_speeds = [calculate_convergence_speed(rewards_matrix[i], threshold=convergence_threshold) for i in range(num_seeds)]
    mean_convergence = np.mean(convergence_speeds)
    
    stabilities = [calculate_stability(rewards_matrix[i])["coeff_of_variation"] for i in range(num_seeds)]
    mean_stability = np.mean(stabilities)
    
    sample_efficiencies = [calculate_sample_efficiency(rewards_matrix[i]) for i in range(num_seeds)]
    mean_se = np.mean(sample_efficiencies)
    
    div_rate = calculate_divergence_rate(rewards_matrix, success_threshold=convergence_threshold)
    
    overfitting_info = [check_overfitting(rewards_matrix[i]) for i in range(num_seeds)]
    overfitting_rate = sum([1 for info in overfitting_info if info["is_overfitting"]]) / num_seeds * 100.0
    
    return {
        "mean_final_reward": float(mean_final_reward),
        "std_final_reward": float(std_final_reward),
        "mean_convergence_speed": float(mean_convergence),
        "mean_stability_cv": float(mean_stability),
        "mean_sample_efficiency_auc": float(mean_se),
        "divergence_rate_pct": float(div_rate),
        "overfitting_rate_pct": float(overfitting_rate)
    }
