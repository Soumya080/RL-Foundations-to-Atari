# 📈 Value Function Approximations: Tabular Q-Learning to Double DQN

When state-action spaces are too large to represent in tabular form, we approximate value functions using neural networks. We analyze **DQN**, the role of **Experience Replay** and **Target Networks**, and the mathematical resolution of **Double DQN**.

---

## 1. Tabular vs. Deep Q-Learning

*   **Tabular Q-Learning:** Stores values in a table $Q(s, a)$. It is limited to low-dimensional, discrete state spaces.
*   **Deep Q-Learning:** Approximates $Q(s, a) \approx Q(s, a; \theta)$ using a neural network parameterised by weights $\theta$. This enables generalization to high-dimensional or continuous states (e.g. raw pixels in Atari games).

---

## 2. The DQN Loss Function

Training a deep network to approximate action-values requires minimizing a sequence of mean-squared Bellman error loss functions:

$$L_i(\theta_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a' \in \mathcal{A}} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i) \right)^2 \right]$$

Where:
*   $\theta_i$ are the parameters of the online Q-network at iteration $i$.
*   $\theta_i^-$ are the parameters of the target Q-network.
*   $\mathcal{D}$ is the experience replay distribution.

---

## 3. Stabilization Mechanisms in DQN

Standard supervised learning assumes data is independent and identically distributed (i.i.d.). RL violates this assumption because consecutive states in a trajectory are highly correlated, and the policy's parameter updates shift the data distribution. DQN solves this using two mechanisms:

### experience Replay ($\mathcal{D}$)
Transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ are collected and stored in a sliding-window buffer. During training, mini-batches are sampled uniformly at random:
$$(s, a, r, s') \sim \text{Uniform}(\mathcal{D})$$
This:
- Breaks temporal correlation of consecutive frames.
- Reuses experience data multiple times for sample efficiency.

### Target Network ($\theta^-$)
Updating parameters $\theta$ that also define the target output (i.e. $y = r + \gamma \max_{a'} Q(s', a'; \theta)$) leads to severe oscillations (the "chasing your own tail" problem).
DQN keeps a separate target network parameterised by $\theta^-$:
- $\theta^-$ is frozen for $N$ steps, then copied: $\theta^- \leftarrow \theta$.
- Alternatively, we use soft target updates: $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ where $\tau \ll 1$.

---

## 4. Double DQN (DDQN)

### The Overestimation Bias Problem
The standard DQN target uses the $\max$ operator:

$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a; \theta^-)$$

Because $Q$-values are noisy approximations, the maximum of a set of noisy variables has a systematic positive bias:

$$\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$$

This overestimation bias propagates back through Bellman backups, leading to suboptimal policies and slow training.

### The Double Q-Learning Target
Double DQN decouples the action selection from the action evaluation:
1.  **Action Selection:** Select the best action using the online network ($\theta$).
2.  **Action Evaluation:** Evaluate this action using the target network ($\theta^-$).

The Double Q target is formulated as:

$$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q\left( S_{t+1}, \arg\max_{a \in \mathcal{A}} Q(S_{t+1}, a; \theta) ; \theta^- \right)$$

This mathematically eliminates the overestimation bias by evaluating the online-greedy action on a distinct set of weights, stabilizing the optimization process.
