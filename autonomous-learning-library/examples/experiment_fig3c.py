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
    timesteps = 200000
    run_experiment(
        [
            '''
                # profiling use jtop
                
                # raw data:
                GPU memory usage for replay buffer size: 
                    fix batchsize=1 [+48:1e6, +6:1e4, 656:1e2]
                    fix batchsize=1024 [+54:1e6, +12:1e4, +6:1e2]
                    fix batchsize=8192 [+74:1e6, +32:1e4, +26:1e2]
            # GPU memory usage for replay buffer size: fix batchsize=1 [656:1e6, 662:1e4, 682:1e2] 
            # GPU memory usage for minibatch size: fix replay=1e4 [656:1, 662:1024, 682:8192]
            '''

            # ddqn.device(DEVICE).name('ddqn;b=1').hyperparameters(minibatch_size=1,replay_buffer_size=10000,replay_start_size=10000),
        ],
        [GymEnvironment('CartPole-v0', DEVICE)],
        timesteps,
        test_episodes=10,
    )
    # plot_returns_100('runs', timesteps=timesteps)


if __name__ == "__main__":
    main()
