'''
Quick example of usage of the run_experiment API.
'''
from all.experiments import run_experiment, plot_returns_100
from all.presets.classic_control import ddqn, dqn, c51
# from all.presets.atari import ddqn, dqn
from all.environments import GymEnvironment, AtariEnvironment


def main():
    # DEVICE = 'cpu'
    DEVICE = 'cuda' # uncomment for gpu support
    timesteps = 40000
    run_experiment(
        [
            # DQN with default hyperparameters
            # dqn.device(DEVICE),
            # # DQN with a custom hyperparameters and a custom name.
            # dqn.device(DEVICE).hyperparameters(replay_buffer_size=100).name('dqn-small-buffer'),
            # # A2C with a custom name
            # a2c.device(DEVICE).name('not-dqn')

            ddqn.device(DEVICE).name('ddqn;b=1').hyperparameters(minibatch_size=1),
            ddqn.device(DEVICE).name('ddqn;b=2').hyperparameters(minibatch_size=2),
            ddqn.device(DEVICE).name('ddqn;b=4').hyperparameters(minibatch_size=4),
            ddqn.device(DEVICE).name('ddqn;b=8').hyperparameters(minibatch_size=8),
            ddqn.device(DEVICE).name('ddqn;b=16').hyperparameters(minibatch_size=16),
            ddqn.device(DEVICE).name('ddqn;b=32').hyperparameters(minibatch_size=32),
            ddqn.device(DEVICE).name('ddqn;b=64').hyperparameters(minibatch_size=64),
            ddqn.device(DEVICE).name('ddqn;b=128').hyperparameters(minibatch_size=128),
            ddqn.device(DEVICE).name('ddqn;b=256').hyperparameters(minibatch_size=256),
            ddqn.device(DEVICE).name('ddqn;b=512').hyperparameters(minibatch_size=512),
            ddqn.device(DEVICE).name('ddqn;b=1024').hyperparameters(minibatch_size=1024),
            ddqn.device(DEVICE).name('ddqn;b=2048').hyperparameters(minibatch_size=2048),

            dqn.device(DEVICE).name('dqn;b=1').hyperparameters(minibatch_size=1),
            dqn.device(DEVICE).name('dqn;b=2').hyperparameters(minibatch_size=2),
            dqn.device(DEVICE).name('dqn;b=4').hyperparameters(minibatch_size=4),
            dqn.device(DEVICE).name('dqn;b=8').hyperparameters(minibatch_size=8),
            dqn.device(DEVICE).name('dqn;b=16').hyperparameters(minibatch_size=16),
            dqn.device(DEVICE).name('dqn;b=32').hyperparameters(minibatch_size=32),
            dqn.device(DEVICE).name('dqn;b=64').hyperparameters(minibatch_size=64),
            dqn.device(DEVICE).name('dqn;b=128').hyperparameters(minibatch_size=128),
            dqn.device(DEVICE).name('dqn;b=256').hyperparameters(minibatch_size=256),
            dqn.device(DEVICE).name('dqn;b=512').hyperparameters(minibatch_size=512),
            dqn.device(DEVICE).name('dqn;b=1024').hyperparameters(minibatch_size=1024),
            dqn.device(DEVICE).name('dqn;b=2048').hyperparameters(minibatch_size=2048),

            c51.device(DEVICE).name('c51;b=1').hyperparameters(minibatch_size=1),
            c51.device(DEVICE).name('c51;b=2').hyperparameters(minibatch_size=2),
            c51.device(DEVICE).name('c51;b=4').hyperparameters(minibatch_size=4),
            c51.device(DEVICE).name('c51;b=8').hyperparameters(minibatch_size=8),
            c51.device(DEVICE).name('c51;b=16').hyperparameters(minibatch_size=16),
            c51.device(DEVICE).name('c51;b=32').hyperparameters(minibatch_size=32),
            c51.device(DEVICE).name('c51;b=64').hyperparameters(minibatch_size=64),
            c51.device(DEVICE).name('c51;b=128').hyperparameters(minibatch_size=128),
            c51.device(DEVICE).name('c51;b=256').hyperparameters(minibatch_size=256),
            c51.device(DEVICE).name('c51;b=512').hyperparameters(minibatch_size=512),
            c51.device(DEVICE).name('c51;b=1024').hyperparameters(minibatch_size=1024),
            c51.device(DEVICE).name('c51;b=2048').hyperparameters(minibatch_size=2048),
        ],
        # [GymEnvironment('CartPole-v0', DEVICE), GymEnvironment('Acrobot-v1', DEVICE)],
        # [GymEnvironment('Acrobot-v1', DEVICE)],

        [GymEnvironment('CartPole-v0', DEVICE)],
        # [AtariEnvironment(env, device='cuda') for env in ['BeamRider', 'Breakout', 'Pong', 'Qbert', 'SpaceInvaders']],
        # [AtariEnvironment(env, device='cuda') for env in ['Pong']],
        timesteps,
        test_episodes=10,
    )
    plot_returns_100('runs', timesteps=timesteps)


if __name__ == "__main__":
    main()
