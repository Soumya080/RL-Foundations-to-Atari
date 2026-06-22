# 🔄 Iterative Policy Evaluation & Contraction Mapping

To compute the state-value function $v_\pi(s)$ of a policy $\pi$, we turn the Bellman Expectation Equation into an iterative algorithm. We formalize this process using **Operator Theory** and prove convergence using the **Contraction Mapping Theorem**.

---

## 1. The Iterative Policy Evaluation Algorithm

Given a known model of the MDP $(\mathcal{P}, \mathcal{R})$, iterative policy evaluation starts with an arbitrary initial value function approximation $V_0 \in \mathbb{R}^{|\mathcal{S}|}$ (typically $V_0(s) = 0$ for all $s$). At each step $k$, the new approximation $V_{k+1}$ is computed from $V_k$ by applying the Bellman Expectation backup across all states:

$$V_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V_k(s') \right], \quad \forall s \in \mathcal{S}$$

---

## 2. The Bellman Expectation Operator ($T^\pi$)

Let $\mathcal{V} = \mathbb{R}^{|\mathcal{S}|}$ be the vector space of all value functions. We define the **Bellman Expectation Operator** $T^\pi: \mathcal{V} \rightarrow \mathcal{V}$ as:

$$(T^\pi V)(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V(s') \right]$$

Iterative Policy Evaluation is simply the repeated application of $T^\pi$:

$$V_{k+1} = T^\pi V_k \implies V_k = (T^\pi)^k V_0$$

The true value function $v_\pi$ is a **fixed point** of this operator:

$$T^\pi v_\pi = v_\pi$$

---

## 3. Proof: $T^\pi$ is a Contraction Mapping

To guarantee that $V_k$ converges to $v_\pi$ from *any* initial vector $V_0$, we prove that $T^\pi$ is a contraction mapping under the infinity norm (maximum norm) $\| \cdot \|_\infty$, defined as:

$$\| U - V \|_\infty = \max_{s \in \mathcal{S}} | U(s) - V(s) |$$

### Theorem
For any $\gamma < 1$, the operator $T^\pi$ is a $\gamma$-contraction under the infinity norm:

$$\| T^\pi U - T^\pi V \|_\infty \le \gamma \| U - V \|_\infty$$

### Proof
For any state $s \in \mathcal{S}$:

$$| (T^\pi U)(s) - (T^\pi V)(s) | = \left| \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a) + \gamma U(s') \right] - \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a) + \gamma V(s') \right] \right|$$

Grouping the terms inside the summation:

$$= \gamma \left| \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left( U(s') - V(s') \right) \right|$$

Using the triangle inequality:

$$\le \gamma \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) | U(s') - V(s') |$$

By definition of the infinity norm, $| U(s') - V(s') | \le \| U - V \|_\infty$ for all $s'$. We substitute this upper bound:

$$\le \gamma \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \| U - V \|_\infty$$

Since $\| U - V \|_\infty$ is constant with respect to the summation variables:

$$= \gamma \| U - V \|_\infty \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a)$$

Since $\sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) = 1$ and $\sum_{a \in \mathcal{A}} \pi(a \mid s) = 1$:

$$= \gamma \| U - V \|_\infty$$

Taking the maximum over all $s \in \mathcal{S}$ on the left-hand side yields:

$$\| T^\pi U - T^\pi V \|_\infty \le \gamma \| U - V \|_\infty$$

$$\tag*{$\blacksquare$}$$

---

## 4. Convergence via the Banach Fixed-Point Theorem

By the **Banach Fixed-Point Theorem**, any contraction mapping on a complete metric space (which $(\mathbb{R}^{|\mathcal{S}|}, \| \cdot \|_\infty)$ is):

1.  Possesses a **unique fixed point** $v_\pi \in \mathcal{V}$ such that $T^\pi v_\pi = v_\pi$.
2.  Iterating $V_{k+1} = T^\pi V_k$ starting from any initial $V_0$ is guaranteed to converge to this unique fixed point:
    $$\lim_{k \rightarrow \infty} V_k = v_\pi$$
3.  The rate of convergence is exponential, bounded by the discount factor $\gamma$:
    $$\| V_k - v_\pi \|_\infty \le \gamma^k \| V_0 - v_\pi \|_\infty$$
