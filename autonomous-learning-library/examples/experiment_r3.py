'''
Quick example of usage of the run_experiment API.
'''

import random

from all.experiments import run_experiment
from all.presets.classic_control import c51, ddqn, dqn
from all.environments import GymEnvironment


def main(algorithm, deadline, batch_size):
    random.seed(1224)
    free_mem_bytes = 1e9 # 1GB
    DEVICE = 'cuda'
    timesteps = 4e4
    data_budget = 64*timesteps
    run_experiment(
        agents=[
            algorithm.device(DEVICE),
        ],
        envs=[GymEnvironment('CartPole-v0', DEVICE)],
        frames=timesteps,
        D_deadline=deadline,
        b_base_batch=batch_size,
        m_batch=400e6,  # 400 MB
        m_base_batch=200e6, # 200 MB
        m_replay=int(600e6),    # 600 MB
        min_replay_size=int(1e4),   # 10K
        max_replay_size=int(2e4),   # 10K
        scaling_factor_batch=1.5,
        gamma_coordinator=4,        # 4 episodes
        total_memory_coordinator=free_mem_bytes,    # 1GB
        coarse_grained=False,
        data_budget=data_budget,
    )


if __name__ == "__main__":
    main(c51, 56.05, 128)
    main(dqn, 65.56, 64)
    # main(ddqn, 158.288)
