from timeit import default_timer as timer
import numpy as np
from all.logging import ExperimentLogger, CometLogger
from all.r3.replay_buffer_size import replay_control

from .experiment import Experiment

# R3 Additions
from all.r3.deadline import Deadline
from all.r3.runtime_coordinator import RuntimeCoordinator
from all.r3.batch_size_control import BatchControl

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

        # R3 related members
        self._agent_has_minibatch_size = hasattr(self._agent, "minibatch_size")
        self._agent_has_replay_buffer = hasattr(self._agent, "replay_buffer")
        self._M = total_memory_coordinator
        self._M_batch = m_batch
        self._M_base = m_base_batch
        self._D = D_deadline
        self._M_replay = m_replay
        self._b_base_batch = b_base_batch
        self._coarse_grained = coarse_grained
        self._current_batch_size = b_base_batch
        self._min_replay_size = min_replay_size
        self._max_replay_size = max_replay_size

        self._runtime_coordinator = RuntimeCoordinator(gamma=gamma_coordinator, total_memory=total_memory_coordinator)
        self._batch_control = BatchControl(min_batch_size=b_base_batch, b_base=b_base_batch, M_base=self._M_base,
                                           scaling_factor=scaling_factor_batch)

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
        # Set the new batch size for the episode
        if self._agent_has_minibatch_size:
            if self._agent_has_replay_buffer:
                self._agent.replay_buffer.minibatch_size = self._current_batch_size

        # Calculate the replay buffer size
        replay_buffer_size_episode = 0
        if self._agent_has_replay_buffer:
            replay_buffer_size_episode = replay_control(agent=self._agent,
                                                        M_replay=self._M_replay,
                                                        min_replay_size=self._min_replay_size,
                                                        max_replay_size=self._max_replay_size)

        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # initialize the episode
        state = self._env.reset()
        action, isTraining = self._agent.act(state)
        returns = 0

        training_time_elapsed = 0

        if self._coarse_grained:
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
                    self._spent_budget += self._current_batch_size
        else:
            # loop until the episode is finished
            step_count = 0
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
                    self._spent_budget += self._current_batch_size

                # Calculate time spent
                time_budget_spent = self._elapsed_train_time / self._D

                # Calculate spend budget
                data_budget_spent = self._spent_budget / self._data_budget

                # Update batch size
                self._current_batch_size = self._batch_control.batch_size_control(M_batch=self._M_batch,
                                                                                  current_batch_size=self._current_batch_size,
                                                                                  time_budget=time_budget_spent,
                                                                                  data_budget=data_budget_spent)
                self._agent.replay_buffer.minibatch_size = self._current_batch_size

                step_count += 1

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)
        training_fps = (self._frame - start_frame) / training_time_elapsed * self._agent.minibatch_size

        self._elapsed_train_time += training_time_elapsed

        # Calculate the latency
        latency = training_time_elapsed

        # log the results
        self._log_training_episode(returns, fps, training_fps, self._elapsed_train_time, self._spent_budget,
                                   self._data_budget, latency=latency, deadline=0.5, rb_size=replay_buffer_size_episode,
                                   b_size=self._current_batch_size)

        # Calculate time spent
        time_budget_spent = self._elapsed_train_time / self._D

        # Calculate spend budget
        data_budget_spent = self._spent_budget / self._data_budget

        # Update batch size
        self._current_batch_size = self._batch_control\
            .batch_size_control(M_batch=self._M_batch,
                                current_batch_size=self._current_batch_size,
                                time_budget=time_budget_spent,
                                data_budget=data_budget_spent)

        # Get updated batch size and replay buffer size
        self._M_batch, self._M_replay = self._runtime_coordinator.\
            update(current_elapsed_training_time=training_time_elapsed,
                   current_episode_rewards=returns,
                   M_batch=self._M_batch,
                   M_replay=self._M_replay)

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

    def close(self):
        if self._agent_has_replay_buffer:
            # Exit the thread.
            self._agent.replay_buffer.stop_thread()
        super().close()

    def _make_logger(self, logdir, agent_name, env_name, verbose, logger):
        if logger == "comet":
            return CometLogger(self, agent_name, env_name, verbose=verbose, logdir=logdir)
        return ExperimentLogger(self, agent_name, env_name, verbose=verbose, logdir=logdir)
