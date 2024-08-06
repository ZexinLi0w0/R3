from timeit import default_timer as timer
import numpy as np
from all.logging import ExperimentLogger, CometLogger

from .experiment import Experiment


class SingleEnvExperiment(Experiment):
    '''An Experiment object for training and testing agents that interact with one environment at a time.'''

    def __init__(
            self,
            preset,
            env,
            data_budget,
            name=None,
            train_steps=float('inf'),
            logdir='runs',
            quiet=False,
            render=False,
            verbose=True,
            logger="tensorboard"
    ):
        self._name = name if name is not None else preset.name
        super().__init__(self._make_logger(logdir, self._name, env.name, verbose, logger), quiet)
        self._logdir = logdir
        self._preset = preset
        self._agent = self._preset.agent(logger=self._logger, train_steps=train_steps)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1
        self._elapsed_train_time = 0
        self._data_budget = data_budget
        self._spent_budget = 0

        if render:
            self._env.render(mode="human")

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf):
        while not self._done(frames, episodes):
            self._run_training_episode()

    def test(self, episodes=100):
        test_agent = self._preset.test_agent()
        returns = []
        for episode in range(episodes):
            episode_return = self._run_test_episode(test_agent)
            returns.append(episode_return)
            self._log_test_episode(episode, episode_return)
        self._log_test(returns)
        return returns

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # initialize the episode
        state = self._env.reset()
        action, isTraining = self._agent.act(state)
        returns = 0

        training_time_elapsed = 0
        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            training_begin = timer()
            action, isTraining = self._agent.act(state)
            training_end = timer()

            training_time_elapsed += (training_end - training_begin)
            returns += state.reward
            self._frame += 1
            # Hack: update the spent budget per timestep
            if isTraining:
                self._spent_budget += self._agent.minibatch_size

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)
        training_fps = (self._frame - start_frame) / training_time_elapsed * self._agent.minibatch_size

        self._elapsed_train_time += training_time_elapsed

        # log the results
        self._log_training_episode(returns, fps, training_fps, self._elapsed_train_time, self._spent_budget, self._data_budget)

        # update experiment state
        self._episode += 1

    def _run_test_episode(self, test_agent):
        # initialize the episode
        state = self._env.reset()
        action = test_agent.act(state)
        returns = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = test_agent.act(state)
            returns += state.reward

        return returns

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes or self._spent_budget >= self._data_budget

    def _make_logger(self, logdir, agent_name, env_name, verbose, logger):
        if logger == "comet":
            return CometLogger(self, agent_name, env_name, verbose=verbose, logdir=logdir)
        return ExperimentLogger(self, agent_name, env_name, verbose=verbose, logdir=logdir)
