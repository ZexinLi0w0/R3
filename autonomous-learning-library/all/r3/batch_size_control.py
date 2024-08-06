import math

class BatchControl(object):

    def __init__(self, min_batch_size=32, b_base=1e6, M_base=1e6, scaling_factor=1.5):
        self._min_batch_size = min_batch_size
        self._max_batch_size = min_batch_size * 4
        self._b_base = b_base
        self._M_base = M_base
        self._scaling_factor = scaling_factor

    def batch_size_control_episode(self, b_base=0.0, d_base=1.0, deadline=0.0, M_batch=0.0, M_base=1.0):
        """
        Function to control the batch size per episode.

        @param b_base - Assigned minibtach size.
        @param d_base - Base deadline value.
        @param deadline - Assigned deadline for this episode.
        @param M_batch - Assigned batch size by the runtime coordinator.
        @param M_base - Average memory consumed by training process for this environment.
        @param env_max_batch - Maximum batch size that can be allocated for this environment.
        @return The newly computed batch size for this episode.
        """

        if d_base == 0 or M_base == 0:
            return self._min_batch_size

        v1 = b_base * deadline / d_base
        v2 = M_batch * b_base / M_base
        print("BATCH SIZE EPISODE: v1: {} v2: {} M_batch: {} b_base: {} d_base: {} M_batch: {} M_base: {}".format(v1, v2, M_batch, b_base, d_base, M_batch, M_base))
        v3 = int(min(v1, v2))

        if v3 < self._min_batch_size:
            return self._min_batch_size
        elif v3 > self._max_batch_size:
            return self._max_batch_size

        return v3

    def batch_size_control_step(self, alpha=0.0, b_episode=0.0, episode_number=1, step_number=1, omega=1.0, b_base=0.0, M_batch=0.0, M_base=1.0):
        """
        Function to control the batch size per episode.

        @param alpha - For controlling the amplitude of the sinusoidal function.
        @param b_base - Assigned minibtach size.
        @param b_episode - Batch size for this episode.
        @param episode_number - The episode number.
        @param step_number - The current step number for this episode.
        @param M_batch - Assigned batch size by the runtime coordinator.
        @param M_base - Average memory consumed by training process for this environment.
        @param env_max_batch - Maximum batch size that can be allocated for this environment.
        @return The newly computed batch size for this step.
        """

        if M_base == 0:
            return self._min_batch_size

        v1 = b_episode * (1 + (alpha * math.sin(omega * step_number)/ episode_number))
        v2 = M_batch * b_base / M_base
        v3 = min(v1, v2)

        if v3 < self._min_batch_size:
            return self._min_batch_size
        elif v3 > self._max_batch_size:
            return self._max_batch_size

        return v3

    def batch_size_control(self, M_batch=0.0, current_batch_size=32, time_budget=1.0, data_budget=1.0):
        if time_budget > data_budget:
            current_batch_size *= self._scaling_factor
        else:
            current_batch_size /= self._scaling_factor

        upper_bound = M_batch * self._b_base / self._M_base

        current_batch_size = min(current_batch_size, upper_bound)

        if current_batch_size > self._max_batch_size:
            current_batch_size = self._max_batch_size
        elif current_batch_size < self._min_batch_size:
            current_batch_size = self._min_batch_size

        return int(current_batch_size)