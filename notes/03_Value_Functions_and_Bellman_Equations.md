# 🧮 Value Functions & Bellman Expectation Equations

In sequential decision-making, we evaluate the quality of a state or action using **Value Functions**. By combining these functions with the recursive return formula, we derive the **Bellman Expectation Equations**.

---

## 1. Value Functions

### State-Value Function ($v_\pi(s)$)
The state-value function $v_\pi(s)$ is the expected return starting from state $s$ and following policy $\pi$ thereafter:

$$v_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]$$

### Action-Value Function ($q_\pi(s, a)$)
The action-value function (or Q-function) $q_\pi(s, a)$ is the expected return starting from state $s$, taking action $a$, and following policy $\pi$ thereafter:

$$q_\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]$$

### Relationship between $v_\pi(s)$ and $q_\pi(s, a)$
The state-value is the expectation of the action-values over the policy's action distribution:

$$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s, a)$$

---

## 2. Derivation of the Bellman Expectation Equation for $v_\pi(s)$

We can expand $v_\pi(s)$ using the law of total expectation and the recursive return structure:

1.  **Start with the definition:**
    $$v_\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]$$
2.  **Substitute the recursive return $G_t = R_{t+1} + \gamma G_{t+1}$:**
    $$v_\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right]$$
3.  **Use the linearity of expectation:**
    $$v_\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} \mid S_t = s \right] + \gamma \mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s \right]$$
4.  **Condition on the action $A_t = a$ and next state $S_{t+1} = s'$:**
    $$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma \mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right] \right]$$
5.  **Notice that $\mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right]$ is exactly the definition of $v_\pi(s')$:**
    $$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma v_\pi(s') \right]$$

This is the **Bellman Expectation Equation for State-Values**. It expresses the value of a state as the immediate expected reward plus the discounted value of the expected next state.

---

## 3. Derivation of the Bellman Expectation Equation for $q_\pi(s, a)$

Similarly, we expand the action-value function:

1.  **Start with the definition:**
    $$q_\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]$$
2.  **Substitute the recursive return:**
    $$q_\pi(s, a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a \right]$$
3.  **Condition on the next state transition $S_{t+1} = s'$:**
    $$q_\pi(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma \mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right] \right]$$
4.  **Rewrite the expectation inside the bracket using the policy $\pi$ at state $s'$:**
    $$\mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right] = v_\pi(s') = \sum_{a' \in \mathcal{A}} \pi(a' \mid s') q_\pi(s', a')$$
5.  **Substitute back to get the final Q-update equation:**
    $$q_\pi(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') q_\pi(s', a') \right]$$

This is the **Bellman Expectation Equation for Action-Values**.
