# 🌐 Markov Decision Processes & Problem Formulation

Reinforcement Learning formally models sequential decision-making under uncertainty using the framework of **Markov Decision Processes (MDPs)**. In this note, we define the mathematical structures that govern agent-environment interactions.

---

## 1. Formal Definition of an MDP

A finite Markov Decision Process is defined as a 5-tuple:

$$\mathcal{M} = \left(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma\right)$$

Where:
*   $\mathcal{S}$ is a finite set of **states** representing all possible configurations of the environment.
*   $\mathcal{A}$ is a finite set of **actions** available to the agent. (Optionally, $\mathcal{A}(s)$ defines actions available from state $s \in \mathcal{S}$).
*   $\mathcal{P}$ is the **state-transition probability function**:
    $$\mathcal{P}(s' \mid s, a) = \mathbb{P}\left(S_{t+1} = s' \mid S_t = s, A_t = a\right)$$
    which defines the probability of transitioning to state $s'$ at time $t+1$, given the agent takes action $a$ in state $s$ at time $t$.
*   $\mathcal{R}$ is the **expected reward function**:
    $$\mathcal{R}(s, a) = \mathbb{E}\left[R_{t+1} \mid S_t = s, A_t = a\right]$$
    or more generally:
    $$\mathcal{R}(s, a, s') = \mathbb{E}\left[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s'\right]$$
    which represents the immediate reward received by the agent after executing action $a$ in state $s$ and transitioning to $s'$.
*   $\gamma \in [0, 1]$ is the **discount factor** weighting immediate versus delayed rewards.

---

## 2. Transition Dynamics

The transition dynamics of the environment are fully characterized by the probability transition kernel $\mathcal{P}$. Since $\mathcal{P}$ defines a probability distribution over the state space for each state-action pair, it satisfies the normalization condition:

$$\sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) = 1, \quad \forall s \in \mathcal{S}, \, a \in \mathcal{A}$$

In model-based settings (e.g., Dynamic Programming), the agent has complete access to $\mathcal{P}$ and $\mathcal{R}$. In model-free settings (e.g., Q-Learning, SARSA), the agent does not know $\mathcal{P}$ or $\mathcal{R}$ and must learn through sampling trajectories.

---

## 3. Policy Definition

A **policy** $\pi$ defines the behavior of the agent by mapping states to actions.

### Stochastic Policy
A stochastic policy $\pi(a \mid s)$ defines a probability distribution over actions given a state:
$$\pi(a \mid s) = \mathbb{P}\left(A_t = a \mid S_t = s\right)$$
satisfying:
$$\sum_{a \in \mathcal{A}} \pi(a \mid s) = 1, \quad \forall s \in \mathcal{S}$$

### Deterministic Policy
A deterministic policy $\mu(s)$ maps each state directly to a single action:
$$\mu(s) = a$$
which can be viewed as a degenerate stochastic policy where $\pi(\mu(s) \mid s) = 1$ and $\pi(a \mid s) = 0$ for all $a \neq \mu(s)$.

---

## 4. The Agent-Environment Interaction Loop

The sequence of states, actions, and rewards forms a stochastic process known as a **trajectory** or **episode**:

$$\tau = \left(S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots, S_{T-1}, A_{T-1}, R_T, S_T\right)$$

At each discrete time step $t = 0, 1, 2, \dots$:
1. The agent observes the current state $S_t \in \mathcal{S}$.
2. The agent selects an action $A_t \sim \pi(\cdot \mid S_t)$.
3. The environment receives action $A_t$, transitions to a new state $S_{t+1} \sim \mathcal{P}(\cdot \mid S_t, A_t)$, and emits an immediate reward $R_{t+1}$ drawn from the reward distribution.
4. The loop repeats.
