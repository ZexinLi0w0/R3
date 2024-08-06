'''
Quick example of usage of the run_experiment API.
'''
import all.presets.atari
from all.experiments import run_experiment
# from all.presets.classic_control import a2c, c51, ddqn, dqn, ppo, rainbow, vac, vpg, vqn, vsarsa
from all.environments import GymEnvironment, AtariEnvironment

def run_algorithm(alg, env_name='CartPole-v0', ts=1e5):
    DEVICE = 'cuda'

    timesteps = ts
    data_budget = 0
    if env_name == 'Breakout':
        timesteps = 1e7
        data_budget = 32*timesteps/4
        env = AtariEnvironment(env_name, device=DEVICE)
        preset = all.presets.atari
        if alg == 'c51':
            alg = preset.c51
        elif alg == 'ddqn':
            alg = preset.ddqn
        elif alg == 'dqn':
            alg = preset.dqn
        else:
            raise ValueError("Unknown algorithm: {}".format(alg))
        alg = alg.device(DEVICE)
    elif env_name == 'CartPole-v0':
        timesteps = 4e4
        data_budget = 64*timesteps
        env = GymEnvironment(env_name, device=DEVICE)
        preset = all.presets.classic_control
        if alg == 'c51':
            alg = preset.c51
        elif alg == 'ddqn':
            alg = preset.ddqn
        elif alg == 'dqn':
            alg = preset.dqn
        else:
            raise ValueError("Unknown algorithm: {}".format(alg))
        alg = alg.device(DEVICE)
    else:
        raise ValueError("Unknown environment: {}".format(env_name))

    run_experiment(
        [alg],
        [env],
        frames=timesteps,
        data_budget=data_budget,
    )


if __name__ == "__main__":
    algorithms = [
        "c51",
        "ddqn",
        "dqn",
    ]

    env_name = [
        'CartPole-v0',
        'Breakout',
    ]

    # For each evironment and algorithm, run an experiment
    for name in env_name:
        for algorithm in algorithms:
            # print("Environment: {}, Algorithm: {}".format(name, algorithm.default_name))
            print("Environment: {}, Algorithm: {}".format(name, algorithm))
            run_algorithm(algorithm, name)
