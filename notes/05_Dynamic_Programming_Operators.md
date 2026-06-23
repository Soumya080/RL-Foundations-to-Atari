# 🏛️ Dynamic Programming Operators & Policy Optimization

Using dynamic programming operators, we can optimize policies. We analyze **Policy Iteration** (built on the **Policy Improvement Theorem**) and **Value Iteration** (derived from the **Bellman Optimality Operator**).

---

## 1. The Policy Improvement Theorem

The foundation of policy iteration is the guarantee that making a policy greedy with respect to its own value function always leads to a strictly better (or equal) policy.

### Theorem
Let $\pi$ and $\pi'$ be a pair of deterministic policies such that, for all $s \in \mathcal{S}$:

$$q_\pi(s, \pi'(s)) \ge v_\pi(s)$$

Then the policy $\pi'$ must obtain a state-value function $v_{\pi'}(s)$ that is greater than or equal to $v_\pi(s)$ across all states:

$$v_{\pi'}(s) \ge v_\pi(s), \quad \forall s \in \mathcal{S}$$

### Mathematical Proof
For any state $s \in \mathcal{S}$:

$$v_\pi(s) \le q_\pi(s, \pi'(s))$$

Expanding the action-value function $q_\pi(s, \pi'(s))$ using the Bellman expectation relation:

$$= \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s \right]$$

Now, we recursively apply the inequality $v_\pi(S_{t+k}) \le q_\pi(S_{t+k}, \pi'(S_{t+k}))$ under the expectation of $\pi'$:

$$\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t = s \right]$$

Expanding the expectation of $q_\pi(S_{t+1}, \pi'(S_{t+1}))$:

$$= \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma \mathbb{E}_{\pi'} \left[ R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1} \right] \mid S_t = s \right]$$

$$= \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t = s \right]$$

Repeating this recursive substitution $n$ times:

$$\le \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n v_\pi(S_{t+n}) \mid S_t = s \right]$$

Taking the limit as $n \rightarrow \infty$, and noting that $\lim_{n\rightarrow\infty} \gamma^n v_\pi(S_{t+n}) = 0$ since $\gamma < 1$:

$$\le \mathbb{E}_{\pi'} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s \right] = v_{\pi'}(s)$$

$$\text{Therefore, } v_{\pi'}(s) \ge v_\pi(s)$$

$$\tag*{$\blacksquare$}$$

---

## 2. Policy Iteration

Policy Iteration alternates between two operations until convergence:

$$\pi_0 \xrightarrow{\text{Evaluation}} v_{\pi_0} \xrightarrow{\text{Improvement}} \pi_1 \xrightarrow{\text{Evaluation}} v_{\pi_1} \xrightarrow{\text{Improvement}} \dots \xrightarrow{\text{Improvement}} \pi^* \xrightarrow{\text{Evaluation}} v^*$$

1.  **Policy Evaluation:** Solve for the value function of the current policy:
    $$V = (I - \gamma \mathcal{P}^\pi)^{-1} \mathcal{R}^\pi$$
2.  **Policy Improvement:** Extract a greedy policy with respect to $V$:
    $$\pi_{new}(s) = \arg\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V(s') \right]$$

If $\pi_{new}(s) == \pi_{old}(s)$ for all $s$, then the policy is stable and represents the optimal policy $\pi^*$.

---

## 3. The Bellman Optimality Operator ($T^*$)

Let $\mathcal{V} = \mathbb{R}^{|\mathcal{S}|}$. We define the **Bellman Optimality Operator** $T^*: \mathcal{V} \rightarrow \mathcal{V}$ as:

$$(T^* V)(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V(s') \right]$$

The optimal value function $v^*$ is the unique fixed point of the optimality operator:

$$T^* v^* = v^*$$

### Contraction Properties
Just like the expectation operator $T^\pi$, the optimality operator $T^*$ is a $\gamma$-contraction mapping under the infinity norm:

$$\| T^* U - T^* V \|_\infty \le \gamma \| U - V \|_\infty$$

---

## 4. Value Iteration

Value Iteration updates the value function directly by applying the Bellman Optimality Operator iteratively:

$$V_{k+1} = T^* V_k$$

Or element-wise:

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V_k(s') \right]$$

### Policy Extraction
Once the value function has converged ($\|V_{k+1} - V_k\|_\infty < \theta$), we extract the optimal policy $\pi^*$ greedily:

$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V(s') \right]$$
