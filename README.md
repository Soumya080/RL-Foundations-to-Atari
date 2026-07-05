# 🎓 Reinforcement Learning Showcase: Foundations to Atari Scaling

This repository is a structured, mathematically rigorous portfolio showcasing implementations of core Reinforcement Learning (RL) algorithms, spanning from classical tabular methods to deep value-based and policy gradient architectures.

Each milestone is structured with:
1. **Mathematical Theory:** LaTeX formulations of Bellman backups, loss functions, and optimization targets.
2. **From-Scratch Python Implementations:** Clean, commented code using standard libraries (`numpy`, `pytorch`, `matplotlib`).
3. **Research Analysis:** Plots showing learning curves, value grids, or path evaluations verifying the algorithms.

---

## 📂 Repository Layout

```
RL-FULLL/
├── 01_Dynamic_Programming_GridWorld.ipynb   # Policy Evaluation, Policy Iteration, Value Iteration
├── 02_Tabular_RL_CliffWalking.ipynb        # Model-Free Control: SARSA vs. Q-Learning side-by-side
├── 03_DQN_CartPole_Research.ipynb          # Deep Q-Network & Dueling DQN on CartPole-v1
├── 04_Actor_Critic_CartPole.ipynb          # Advantage Actor-Critic (A2C) on CartPole
├── 05_REINFORCE_Framework/                  # Modular Policy Gradient framework on CartPole
│   ├── reinforce_agent.py                  # Agent loop and Policy network
│   ├── experiment_manager.py               # Hyperparameter sweep manager
│   ├── plotting.py                         # Plotting utility for training rewards
│   └── REINFORCE_CartPole.ipynb            # Training entrypoint
├── 06_Atari_DQN_Pong/                       # Visual Deep Q-Learning scaled to Atari Pong (ALE)
│   ├── Setting_the_env.py                  # Frame stacking and preprocessing wrapper
│   └── Atari_DQN_Pong.ipynb                # DQN training and ALE wrappers
└── notes/                                   # Rigorous theoretical study sheets (01-08)
    ├── 01_MDP_and_Problem_Formulation.md   # MDP formal tuple and transition dynamics
    ├── 02_Markov_Property_and_Return.md    # Return recursive definition proof
    ├── 03_Value_Functions_and_Bellman_Equations.md # Bellman Expectation Equation derivations
    ├── 04_Iterative_Policy_Evaluation.md   # Contraction mapping (Banach Fixed-Point) proof
    ├── 05_Dynamic_Programming_Operators.md # Policy Improvement Theorem proof
    ├── 06_Monte_Carlo_Estimation.md        # First vs. Every-Visit MC & Bias-Variance trade-offs
    ├── 07_Dueling_DQN_Nomenclature.md      # Decoupling Q into V and A & Identifiability resolution
    └── 08_Value_Function_Approximations.md # Target networks, replay buffers, and Double DQN
```

---

## 📈 Summary of Algorithms Implemented

### 1. Dynamic Programming in GridWorld
*   **Concepts:** Exact calculation of $v_\pi(s)$ and optimal policy $\pi^*(s)$ under known transitions.
*   **Mathematical updates:**
    *   *Policy Evaluation:* $V_{k+1}(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a)[r + \gamma V_k(s')]$
    *   *Value Iteration:* $V_{k+1}(s) \leftarrow \max_{a} \sum_{s'} \mathcal{P}(s'|s,a)[r + \gamma V_k(s')]$
*   **Finding:** Value Iteration converges in fewer total state sweeps compared to Policy Iteration, though both arrive at the identical optimal control policy.

### 2. Tabular TD Control on Cliff Walking
*   **Concepts:** Model-free learning from experience transitions, highlighting **on-policy** vs. **off-policy** updates.
*   **Mathematical updates:**
    *   *SARSA (On-policy):* $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$
    *   *Q-Learning (Off-policy):* $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a} Q(S', a) - Q(S, A)]$
*   **Finding:** Because SARSA incorporates its $\epsilon$-greedy exploration behavior into its target update, it learns the **safe path** (one row above the cliff) to avoid falling during exploration. Q-learning evaluates greedily, learning the **optimal but risky path** directly along the cliff edge.

### 3. Deep Q-Networks (DQN & Dueling DQN)
*   **Concepts:** Approximating continuous state-value functions using Neural Networks with experience replay and frozen target networks. Resolving overestimation bias.
*   **Mathematical updates:**
    *   *DQN Loss:* $L(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$
    *   *Double Q-Learning target:* $y^{\text{DoubleQ}} = r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\right)$
    *   *Dueling Q-Value:* $Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$

### 4. Policy Gradient (REINFORCE)
*   **Concepts:** Parameterizing the policy directly $\pi_\theta(a|s)$ and performing gradient ascent on expected return using log-likelihood trick and baseline subtraction.
*   **Mathematical updates:**
    *   *Policy Gradient:* $\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t|S_t) (G_t - b(S_t)) \right]$
    *   *Baseline $b(S_t)$:* Typically an running average return or a parameterized state-value baseline to reduce variance without introducing bias.

---

## 🛠️ Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Soumya080/RL-Foundations-to-Atari
   cd RL-FULLL
   ```

2. Install dependencies:
   ```bash
   pip install numpy torch matplotlib gym gymnasium[classic_control]
   ```

3. Run the notebooks:
   ```bash
   jupyter notebook
   ```
   Open any of the numbered `.ipynb` files to review the code implementation, execution traces, and diagnostic plots.
