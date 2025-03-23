from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
import random

# Build on base policy:
class Policy(ActorCriticPolicy):
    def __init__(self, base_policy):
        super().__init__(
            base_policy.observation_space,
            base_policy.action_space,
            lr_schedule=lambda _: 0.0003,
        )
        self.load_state_dict(base_policy.state_dict())

    def forward(self, obs):
        return super().forward(obs)

    def forward_actor(self, obs):
        return super().forward_actor(obs)

    def forward_critic(self, obs):
        return super().forward_critic(obs)

    def predict(self, obs, state, episode_start, deterministic):
        actions, _ = super().predict(obs, state, episode_start, deterministic)

        # Randomly cut engines for fun
        p = 0.01
        sputter = random.random() < p
        sputter = False

        # Convert obs to numpy if it's a torch tensor
        obs_array = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()

        # Handle single observation (shape (obs_dim,))
        if obs_array.ndim == 1:
            if obs_array[6] or obs_array[7] or sputter:
                return 0, None  # Cut engines
        return actions, None

        # Handle batched observation (shape (batch_size, obs_dim))
        for i in range(obs_array.shape[0]):
            if obs_array[i][6] or obs_array[i][7] or sputter:
                actions[i] = 0  # Cut engines for that env
        return actions, None
