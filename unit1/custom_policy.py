from stable_baselines3.common.policies import ActorCriticPolicy

# Build on base policy:
class Policy(ActorCriticPolicy):
	def __init__(self, base_policy):
		super().__init__(base_policy.observation_space, base_policy.action_space, lr_schedule = lambda _ : 0.0003)
		self.load_state_dict(base_policy.state_dict())

	def forward(self, obs):
		return super().forward(obs)

	def forward_actor(self, obs):
		return super().forward_actor(obs)

	def forward_critic(self, obs):
		return super().forward_critic(obs)

	def predict(self, obs, state, episode_start, deterministic):
		action, _ = super().predict(obs, state, episode_start, deterministic)

		# Cut Engines on Touchdown
		if obs[6] or obs[7]:
			return (0, None)
		else:
			return action, None
