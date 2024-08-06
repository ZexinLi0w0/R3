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
        logger="tensorboard",
        D_deadline=0.0,
        b_base_batch=64,
        m_batch=1e6,
        m_base_batch=1e4,
        m_replay=1e4,
        min_replay_size=1e4,
        max_replay_size=1e4,
        scaling_factor_batch=1.5,
        gamma_coordinator=1,
        total_memory_coordinator=0,
        coarse_grained=False,
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
                logger=logger,
                D_deadline=D_deadline,
                b_base_batch=b_base_batch,
                m_batch=m_batch,
                m_base_batch=m_base_batch,
                m_replay=m_replay,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                scaling_factor_batch=scaling_factor_batch,
                gamma_coordinator=gamma_coordinator,
                total_memory_coordinator=total_memory_coordinator,
                coarse_grained=coarse_grained,
            )
            print("experiment information: {}, D: {}, b_base: {}, m_batch: {}, m_base: {}, scaling_factor: {}, "
                  .format(preset, D_deadline, b_base_batch, m_batch, m_base_batch, scaling_factor_batch) +
                  "gamma: {}, total_memory: {}, coarse_gained: {}"
                  .format(gamma_coordinator, total_memory_coordinator, coarse_grained))
            print("Frame: {}, Data Budget: {}".format(frames, data_budget))
            experiment.train(frames=frames)
            experiment.save()
            # experiment.test(episodes=test_episodes)
            experiment.close()


def get_experiment_type(preset):
    if isinstance(preset, ParallelPreset):
        return ParallelEnvExperiment
    return SingleEnvExperiment
