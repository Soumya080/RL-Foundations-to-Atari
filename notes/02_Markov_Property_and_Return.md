# ⛓️ The Markov Property & Discounted Return

To establish a mathematically solvable framework for reinforcement learning, we rely on the **Markov Property** to model state representations and define the optimization objective through **Discounted Returns**.

---

## 1. The Markov Property

A stochastic process has the **Markov Property** if the conditional probability distribution of future states depends only upon the present state, and not on the sequence of events that preceded it.

### Mathematical Formulation
Formally, a state transition $S_t \rightarrow S_{t+1}$ is Markovian if and only if:

$$\mathbb{P}\left(S_{t+1} = s_{t+1} \mid S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, A_{t-1} = a_{t-1}, \dots, S_0 = s_0, A_0 = a_0\right) = \mathbb{P}\left(S_{t+1} = s_{t+1} \mid S_t = s_t, A_t = a_t\right)$$

### Implications
*   **Memoryless Dynamics:** The current state $S_t$ serves as a sufficient statistic of the historical trajectory. Once $S_t$ is observed, the history $(S_0, A_0, \dots, S_{t-1}, A_{t-1})$ provides no additional information about the future.
*   **State Compression:** Practical RL systems must design state representations that compress history while retaining this property (e.g., frame stacking in Atari games or hidden states in recurrent networks).

---

## 2. Expected Cumulative Return ($G_t$)

The goal of the agent is to select actions that maximize the expected long-term cumulative reward. We define the **Return** $G_t$ at time step $t$ as the discounted sum of future rewards:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Where:
*   $R_{t+k+1}$ is the reward received at step $t+k+1$.
*   $\gamma \in [0, 1]$ is the **discount factor**.

### The Role of the Discount Factor ($\gamma$)
1.  **Mathematical Convergence:** For infinite-horizon environments ($T = \infty$), the return would diverge to infinity if $\gamma = 1$. When $\gamma < 1$ and rewards are bounded ($|R_t| \le R_{max}$), the return is guaranteed to converge:
    $$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \le \sum_{k=0}^{\infty} \gamma^k R_{max} = \frac{R_{max}}{1 - \gamma}$$
2.  **Temporal Preference:** $\gamma$ acts as a dial between greediness and planning:
    *   $\gamma \rightarrow 0$: The agent behaves **myopically** (greedy, only optimizing immediate reward $R_{t+1}$).
    *   $\gamma \rightarrow 1$: The agent becomes **far-sighted** (highly strategic, valuing long-term outcomes equally to immediate feedback).
3.  **Uncertainty Modeling:** Discounting mathematically represents the probability of sudden episode termination or future variance.

---

## 3. Proof: Recursive Structure of the Return

One of the most important derivations in Reinforcement Learning is the recursive representation of the return, which enables dynamic programming and temporal difference backups.

### Derivation
Starting from the definition of $G_t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \dots$$

Factoring out the discount parameter $\gamma$ from all terms past the first:

$$G_t = R_{t+1} + \gamma \left( R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \dots \right)$$

Observe that the expression inside the parentheses is exactly the return definition evaluated at time step $t+1$:

$$G_{t+1} = R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \dots$$

Substituting this back into the factored equation yields:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

### Significance
This recursive relation shows that the return at the current step is equal to the immediate reward plus the discounted return of the next step. This allows us to estimate returns iteratively without waiting until the end of an infinite trajectory.
