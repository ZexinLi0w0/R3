class RuntimeCoordinator(object):

	def __init__(self, gamma=2, total_memory=1):
		self.gamma = gamma
		self.M = total_memory
		self.episode_rewards = []
		self.episode_training_times = []

	def coordinate(self, M_batch=1024, M_replay=10000, episode_number=1, episode_runtime=0.0, deadline=1.0, cummulative_reward=0.0):
		"""
		Function to update the batch size and the replay buffer size
		based on equations (6), (7), (8) and (9)

		@param M_batch - Batch size currently being used.
		@param M_replay - Replay buffer size currently being used.
		@param episode_runtime - Current latency for running the episode.
		@param deadline - The deadline assigned for that episode.
		@param cummulative_reward - The cummulative reward for current episode.
		@param training_loss - Training loss for the current episode.
		@return A tuple containing the updated batch and replay buffer
				sizes.
		"""

		# Calculate cumulative reward for episode gamma
		# and save the reward for later use. Pop the reward
		# from the list only when the number of rewards saved
		# is gamma.
		cummulative_reward_gamma = 1.0
		self.episode_rewards.append(cummulative_reward)
		if episode_number > self.gamma:
			cummulative_reward_gamma = self.episode_rewards[0]
			self.episode_rewards = self.episode_rewards[1:]

		# Ratio of current runtime vs. subtask deadline
		alpha = episode_runtime / deadline

		# Ratio of cummulative rewards
		if cummulative_reward_gamma == 0.0:
			cummulative_reward_gamma = 1.0
		beta = cummulative_reward / cummulative_reward_gamma

		print("RUNTIME COORDINATOR: ALPHA: {}, BETA: {}".format(alpha, beta))

		# Intermediate calculation
		M_batch_new = M_batch * (1 + (max(alpha - 1, 0) * (1 - min(beta, 1))))
		M_replay_new = M_replay * (1 + (min(alpha, 1) * max(1 - beta, 0)))

		# Final calculations for scaling
		M_batch_final = self.M * M_batch_new / (M_batch_new + M_replay_new)
		M_replay_final = self.M * M_replay_new / (M_batch_new + M_replay_new)
		return (M_batch_final, M_replay_final)

	def update(self, current_elapsed_training_time=0.01, current_episode_rewards=0.001, M_batch=1024, M_replay=10000):
		
		def _calculate_average(l, v):
			history_len = len(l)

			if history_len == 0:
				return 0

			history_sum = sum(l)
			numerator = v * history_len
			denominator = history_sum
			return (numerator / denominator)

		alpha = _calculate_average(self.episode_training_times, current_elapsed_training_time)
		beta = _calculate_average(self.episode_rewards, current_episode_rewards)

		if len(self.episode_rewards) == self.gamma:
			self.episode_rewards = self.episode_rewards[1:]
			self.episode_training_times = self.episode_training_times[1:]

		self.episode_rewards.append(current_episode_rewards)
		self.episode_training_times.append(current_elapsed_training_time)

		# Intermediate calculation
		M_batch_new = M_batch * (1 + (max(alpha - 1, 0) * (1 - min(beta, 1))))
		M_replay_new = M_replay * (1 + (min(alpha, 1) * max(1 - beta, 0)))

		# Final calculations for scaling
		M_batch_final = self.M * M_batch_new / (M_batch_new + M_replay_new)
		M_replay_final = self.M * M_replay_new / (M_batch_new + M_replay_new)
		return (M_batch_final, M_replay_final)
