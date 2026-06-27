# 🧠 Dueling DQN Architecture & Identifiability

Deep Q-Networks (DQN) map states to action-values. However, in many states, the value of the state is independent of which action is selected. **Dueling DQN** solves this by decoupling state value and action advantage.

---

## 1. Limitations of Standard DQN

A standard DQN output layer directly approximates the action-value function:

$$Q(s, a; \theta) \approx q_*(s, a)$$

For states where multiple actions lead to similar outcomes (e.g. driving on an empty highway, where moving left or right has little impact compared to keeping speed), standard DQN must learn the action-values for each action independently. This is sample-inefficient because the network updates values for actions it doesn't choose, wasting gradient steps.

---

## 2. Decoupling Value and Advantage

Dueling DQN decomposes the Q-function into two streams:
1.  **State-Value Stream ($V(s)$):** Estimates the overall value of being in state $s$.
2.  **Advantage Stream ($A(s, a)$):** Estimates the relative importance of action $a$ compared to other actions in state $s$.

$$\text{Q-Value definition: } q(s, a) = v(s) + a(s, a)$$

Mathematically, the advantage function is defined as:

$$a(s, a) = q(s, a) - v(s)$$

By definition, the expected advantage under the optimal policy is zero:

$$\max_{a \in \mathcal{A}} a^*(s, a) = 0$$

---

## 3. The Identifiability Problem

If we define the network output simply as:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + A(s, a; \theta, \alpha)$$

Where:
*   $\theta$ are shared network parameters.
*   $\beta$ are value stream parameters.
*   $\alpha$ are advantage stream parameters.

The equation is **unidentifiable**. Given $Q(s, a)$, we cannot uniquely reconstruct $V(s)$ and $A(s, a)$. For example, we could add a constant $C$ to $V(s)$ and subtract it from all $A(s, a)$ outputs, resulting in the identical $Q(s, a)$ output. This instability degrades performance during backpropagation.

---

## 4. Mathematical Resolution: Average Subtraction

To enforce uniqueness and identifiability, we force the advantage stream to have a mean of zero (or a maximum of zero) for each state.

### Formulation 1: Max-Subtraction (Strict Definition)
We subtract the maximum advantage output:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \max_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

This maps directly to theory, as $\max_{a'} Q(s, a'; \theta, \alpha, \beta) = V(s; \theta, \beta)$, matching the exact definition of $v(s)$ as the value of the optimal action.

### Formulation 2: Mean-Subtraction (Practical Choice)
In practice, subtracting the mean advantage improves training stability over max-subtraction:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

While this shifts the absolute value of $V(s)$ away from the exact theoretical state-value $v(s)$ (by a constant offset equal to the mean advantage), it stabilizes gradients. This is because the mean changes smoothly across training steps, whereas the maximum action selection can oscillate.
