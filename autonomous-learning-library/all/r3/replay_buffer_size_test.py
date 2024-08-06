from all.presets.classic_control import dqn
from all.environments import GymEnvironment
from all.r3.replay_buffer_size import replay_control

import unittest

class ReplayBufferSizeTest(unittest.TestCase):

    def setUp(self):
        self._agent = dqn.hyperparameters(replay_buffer_size=100).env(GymEnvironment('CartPole-v0', 'cuda')).build().agent()
        assert(hasattr(self._agent, "replay_buffer"))

    def test_no_agent(self):
        assert(replay_control(None) == 0.0)

    def test_default(self):
        assert(replay_control(self._agent) == 10000)

    def test_set_mreplay(self):
        assert(replay_control(self._agent, 1000) == 1000)

    def tearDown(self):
        self._agent.replay_buffer.stop_thread()

if __name__ == "__main__":
    unittest.main()
