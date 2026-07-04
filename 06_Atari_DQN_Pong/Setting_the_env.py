import gymnasium as gym
import ale_py

gym.register_envs(ale_py)  # Register Atari envs with the ALE namespace

env = gym.make(
    "ALE/Pong-v5",
    render_mode="rgb_array"
)

obs, info = env.reset()

print(obs.shape)