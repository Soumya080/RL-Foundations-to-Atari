# 🎲 Monte Carlo Methods & Temporal Difference Prediction

When the environment dynamics $(\mathcal{P}, \mathcal{R})$ are unknown, we must rely on sample trajectories to evaluate and optimize policies. We explore **Monte Carlo (MC) Estimation** and analyze the **Bias-Variance Trade-off** between MC and **Temporal Difference (TD)** learning.

---

## 1. Monte Carlo Prediction

Monte Carlo methods estimate the value function by averaging the returns observed over complete episodes.

### First-Visit Monte Carlo (FVMC)
Estimates $v_\pi(s)$ by averaging returns only from the first time state $s$ is visited within each episode:

$$V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s)$$

Where $G_i(s)$ is the return following the first visit to $s$ in episode $i$, and $N(s)$ is the total number of episodes in which $s$ was visited. By the **Law of Large Numbers**, as $N(s) \rightarrow \infty$, the estimate converges to the expectation:

$$\lim_{N(s) \rightarrow \infty} V(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = v_\pi(s)$$

### Every-Visit Monte Carlo (EVMC)
Averages returns following *every* visit to state $s$ within each episode. While EVMC is biased in small sample sizes due to state correlation within an episode, it converges to the same asymptotic limit and often has lower variance in practice.

---

## 2. Incremental Mean Updates

Instead of keeping all returns in memory, we update the value function running mean incrementally:

$$V_{n+1} = \frac{1}{n} \sum_{i=1}^{n} G_i = V_n + \frac{1}{n}\left(G_n - V_n\right)$$

In non-stationary environments, we replace the step-size parameter $\frac{1}{n}$ with a constant learning rate $\alpha \in (0, 1]$:

$$V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right)$$

This update slides the estimate toward the target return $G_t$.

---

## 3. Bias-Variance Comparison: Monte Carlo vs. Temporal Difference

Temporal Difference (TD) learning updates its values using bootstrapping:

$$V(S_t) \leftarrow V(S_t) + \alpha \left( R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right)$$

This highlights a fundamental distinction in target estimators:

| Feature | Monte Carlo (MC) Target $G_t$ | Temporal Difference (TD) Target $R_{t+1} + \gamma V(S_{t+1})$ |
| :--- | :--- | :--- |
| **Formula** | $G_t = R_{t+1} + \gamma R_{t+2} + \dots$ | $R_{t+1} + \gamma V(S_{t+1})$ |
| **Bias** | **Unbiased:** $\mathbb{E}[G_t \mid S_t = s] = v_\pi(s)$ | **Biased:** Depends on the accuracy of the estimator $V(S_{t+1})$ |
| **Variance** | **High:** Sum of many random transitions and rewards | **Low:** Depends on only one random transition and reward |
| **Updates** | **Offline:** Must wait until episode completion | **Online:** Updates step-by-step (after one transition) |
| **Markovian** | Agnostic to Markov assumption (works on POMDPs) | Assumes Markov property (bootstrapping values) |

### Mathematical Bias Formulation for TD(0)
The TD target can be written as:

$$Y_t^{\text{TD}} = R_{t+1} + \gamma V_k(S_{t+1})$$

The expected TD target is:

$$\mathbb{E}[Y_t^{\text{TD}} \mid S_t = s] = \sum_{a} \pi(a \mid s) \sum_{s'} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a) + \gamma V_k(s') \right] = (T^\pi V_k)(s)$$

Since $V_k$ is generally not equal to $v_\pi$ during training:

$$\mathbb{E}[Y_t^{\text{TD}} \mid S_t = s] \neq v_\pi(s)$$

This introduces approximation **bias**, which shrinks to zero only as $V_k \rightarrow v_\pi$.

### Mathematical Variance Formulation for MC
Let $G_t$ be the sum of discounted rewards. Since each state transition and reward is a random variable, the variance of the MC target accumulates over time:

$$\text{Var}(G_t) = \text{Var}\left( \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \right)$$

If rewards are highly stochastic or trajectories are long, $\text{Var}(G_t)$ is extremely high, requiring a large number of episodes to obtain accurate value estimations.
