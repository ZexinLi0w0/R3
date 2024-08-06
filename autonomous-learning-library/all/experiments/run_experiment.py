from .single_env_experiment import SingleEnvExperiment
from .parallel_env_experiment import ParallelEnvExperiment
from all.presets import ParallelPreset


def run_experiment(
        agents,
        envs,
        frames,
        data_budget=1e6,
        logdir='runs',
        quiet=False,
        render=False,
        test_episodes=100,
        verbose=True,
        logger="tensorboard"
):
    if not isinstance(agents, list):
        agents = [agents]

    if not isinstance(envs, list):
        envs = [envs]

    for env in envs:
        for preset_builder in agents:
            preset = preset_builder.env(env).build()
            make_experiment = get_experiment_type(preset)
            experiment = make_experiment(
                preset,
                env,
                data_budget,
                train_steps=frames,
                logdir=logdir,
                quiet=quiet,
                render=render,
                verbose=verbose,
                logger=logger
            )
            # print("experiment information: {}, minibatch: {}".format(preset, preset_builder._hyperparameters['minibatch_size']))
            print("experiment information: {}".format(preset))
            print("Frame: {}, Data Budget: {}".format(frames, data_budget))
            experiment.train(frames=frames)
            experiment.save()
            # experiment.test(episodes=test_episodes)
            experiment.close()


def get_experiment_type(preset):
    if isinstance(preset, ParallelPreset):
        return ParallelEnvExperiment
    return SingleEnvExperiment
