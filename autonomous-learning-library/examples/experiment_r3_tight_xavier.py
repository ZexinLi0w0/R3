'''
Quick example of usage of the run_experiment API.
'''

import random

from all.experiments import run_experiment
from all.presets.classic_control import c51, ddqn, dqn
from all.profiler.profiler import Profiler
from all.environments import GymEnvironment


def main(algorithm, deadline, batch_size):
    # p = Profiler()
    # _, _, free_mem_mb = p.profile_memory_get_bytes()
    random.seed(1224)
    free_mem_mb = 1e9
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
        m_batch=20e6,
        m_base_batch=1e4,
        scaling_factor_batch=1.5,
        gamma_coordinator=4,
        total_memory_coordinator=free_mem_mb,
        coarse_grained=False,
        data_budget=data_budget,
    )


if __name__ == "__main__":
    main(c51, 210, 128)
    main(dqn, 230, 64)
    # main(ddqn, 158.288)
