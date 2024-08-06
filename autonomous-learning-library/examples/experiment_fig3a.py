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

            dqn.device(DEVICE).name('dqn;b=4;r=100').hyperparameters(minibatch_size=4, replay_buffer_size=100),
            dqn.device(DEVICE).name('dqn;b=16;r=100').hyperparameters(minibatch_size=16, replay_buffer_size=100),
            dqn.device(DEVICE).name('dqn;b=64;r=100').hyperparameters(minibatch_size=64, replay_buffer_size=100),

            dqn.device(DEVICE).name('dqn;b=4;r=1000').hyperparameters(minibatch_size=4, replay_buffer_size=1000),
            dqn.device(DEVICE).name('dqn;b=16;r=1000').hyperparameters(minibatch_size=16, replay_buffer_size=1000),
            dqn.device(DEVICE).name('dqn;b=64;r=1000').hyperparameters(minibatch_size=64, replay_buffer_size=1000),
    
            dqn.device(DEVICE).name('dqn;b=4;r=10000').hyperparameters(minibatch_size=4, replay_buffer_size=10000),
            dqn.device(DEVICE).name('dqn;b=16;r=10000').hyperparameters(minibatch_size=16, replay_buffer_size=10000),
            dqn.device(DEVICE).name('dqn;b=64;r=10000').hyperparameters(minibatch_size=64, replay_buffer_size=10000),
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
