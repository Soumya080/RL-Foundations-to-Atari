import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym

class PolicyNetwork(nn.Module):
    """
    Parametrized Policy Network mapping states to action logits.
    """
    def __init__(self, state_dim=4, action_dim=2, hidden_sizes=[128, 128], activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_size))
            layers.append(activation())
            in_dim = h_size
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

class ValueNetwork(nn.Module):
    """
    State-Value Network used as a baseline to reduce gradient variance.
    """
    def __init__(self, state_dim=4, hidden_sizes=[128, 128], activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_size))
            layers.append(activation())
            in_dim = h_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

class REINFORCEAgent:
    """
    Production-quality REINFORCE agent supporting:
    - Parameterized policy and value architectures.
    - Standard REINFORCE returns.
    - Return normalization.
    - Moving average baseline & Value network baseline.
    - Action-space entropy regularization.
    - Hyperparameter configurations.
    """
    def __init__(
        self,
        state_dim=4,
        action_dim=2,
        lr_policy=1e-3,
        lr_value=1e-3,
        gamma=0.99,
        hidden_sizes=[128, 128],
        normalize_returns=True,
        baseline_type=None,  # Options: None, 'moving_average', 'value_network'
        entropy_coef=0.0,
        reward_scale=1.0,
        clip_grad_norm=10.0,
        device="cpu"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.baseline_type = baseline_type
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.clip_grad_norm = clip_grad_norm

        # Policy Network
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)

        # Baseline value tracking
        self.value_net = None
        self.value_optimizer = None
        if self.baseline_type == "value_network":
            self.value_net = ValueNetwork(state_dim, hidden_sizes).to(self.device)
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr_value)
        elif self.baseline_type == "moving_average":
            self.running_baseline = 0.0
            self.baseline_alpha = 0.1

    def select_action(self, state):
        """
        Select action based on policy network logits.
        Returns: action, log_prob, entropy.
        """
        state_t = torch.FloatTensor(state).to(self.device)
        logits = self.policy_net(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

    def compute_returns(self, rewards):
        """
        Compute discounted returns G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k}
        """
        returns = []
        g = 0.0
        # Iterate backwards to compute cumulative reward
        for r in reversed(rewards):
            scaled_r = r * self.reward_scale
            g = scaled_r + self.gamma * g
            returns.insert(0, g)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, states, rewards, log_probs, entropies):
        """
        Perform a policy (and baseline) update using a collected trajectory.
        """
        # Convert inputs to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        log_probs_t = torch.stack(log_probs).to(self.device)
        entropies_t = torch.stack(entropies).to(self.device)

        returns_t = self.compute_returns(rewards)

        # Initialize advantage
        advantages = returns_t.clone()

        # Apply Baseline
        value_loss_val = 0.0
        if self.baseline_type == "value_network":
            # Value network prediction
            state_values = self.value_net(states_t).squeeze(-1)
            # Baseline-subtracted return (advantage)
            advantages = returns_t - state_values
            
            # Value Network loss (MSE)
            value_loss = F.mse_loss(state_values, returns_t)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.clip_grad_norm)
            self.value_optimizer.step()
            value_loss_val = value_loss.item()
            
            # Detach advantages to prevent backpropagating into value network from policy loss
            advantages = advantages.detach()

        elif self.baseline_type == "moving_average":
            batch_mean = returns_t.mean().item()
            # Advantage is return minus running baseline
            advantages = returns_t - self.running_baseline
            # Update baseline
            self.running_baseline += self.baseline_alpha * (batch_mean - self.running_baseline)

        # Return Normalization
        if self.normalize_returns:
            if len(advantages) > 1:
                std = advantages.std()
                if std > 1e-8:
                    advantages = (advantages - advantages.mean()) / std
                else:
                    advantages = advantages - advantages.mean()
            else:
                advantages = advantages - advantages.mean()

        # Policy Loss computation: negative log prob weighted by advantage minus entropy bonus
        policy_loss = - (log_probs_t * advantages + self.entropy_coef * entropies_t).mean()

        # Policy Network Update
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # Track gradient norm before step
        grad_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad_norm)
            
        self.policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_val,
            "grad_norm": grad_norm,
            "mean_entropy": entropies_t.mean().item()
        }

def train_agent(
    env,
    agent,
    num_episodes=500,
    max_steps_per_episode=500,
    logger=None
):
    """
    Run the interaction loop for CartPole-v1.
    """
    episode_rewards = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    grad_norms = []
    entropies = []

    for ep in range(num_episodes):
        state, info = env.reset()
        
        states = []
        rewards = []
        log_probs = []
        action_entropies = []
        
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            action, log_prob, entropy = agent.select_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)
            action_entropies.append(entropy)

            state = next_state
            total_reward += reward
            steps += 1

        # Train policy
        update_info = agent.update(states, rewards, log_probs, action_entropies)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        policy_losses.append(update_info["policy_loss"])
        value_losses.append(update_info["value_loss"])
        grad_norms.append(update_info["grad_norm"])
        entropies.append(update_info["mean_entropy"])

        if logger is not None and (ep + 1) % 50 == 0:
            logger.info(f"Episode {ep+1:04d} | Return: {total_reward:.1f} | Loss: {update_info['policy_loss']:.4f} | Entropy: {update_info['mean_entropy']:.3f}")

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "grad_norms": grad_norms,
        "entropies": entropies
    }
