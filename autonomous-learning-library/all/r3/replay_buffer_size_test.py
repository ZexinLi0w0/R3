import unittest

from all.environments import GymEnvironment
from all.presets.classic_control import dqn
from all.r3.replay_buffer_size import replay_control


class ReplayBufferSizeTest(unittest.TestCase):
    """Tests for ``replay_control`` on top of ALL 0.9.1.

    Notes for reviewers
    -------------------
    * The legacy v1.0.0 tests asserted ``replay_control(self._agent) == 10000``
      and ``replay_control(self._agent, 1000) == 1000`` against the *default*
      ``min_replay_size=5e5`` / ``max_replay_size=1e6`` bounds.  Those bounds
      necessarily clamp the result to ``min_replay_size`` (500_000), so the
      legacy assertions could never pass with the legacy implementation either
      -- they were latent bugs that went unnoticed because the test suite was
      not wired into CI before PR #1.
    * The R3 paper's actual experiments (see legacy
      ``examples/experiment_r3.py``) use ``min_replay_size=1e4`` /
      ``max_replay_size=2e4``, so we exercise ``replay_control`` with bounds
      consistent with that real-world usage and additionally cover the clamp
      paths explicitly.
    """

    def setUp(self):
        device = "cuda"
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:  # pragma: no cover - torch is a hard dep
            device = "cpu"

        self._agent = (
            dqn.hyperparameters(replay_buffer_size=100)
            .env(GymEnvironment("CartPole-v0", device))
            .build()
            .agent()
        )
        assert hasattr(self._agent, "replay_buffer")

    def tearDown(self):
        # ALL 0.9.1's ExperienceReplayBuffer is synchronous; ``stop_thread`` is
        # a no-op kept for R3 API compatibility.  Calling it here exercises the
        # shim and matches the original legacy teardown.
        self._agent.replay_buffer.stop_thread()

    def test_no_agent(self):
        self.assertEqual(replay_control(None), 0.0)

    def test_default_clamps_to_min(self):
        # With M_replay=10000 (default) and state_size=1 (empty buffer), the
        # raw new_size is 10_000 which is below the default min_replay_size of
        # 5e5, so the result must be clamped up to 500_000.
        self.assertEqual(replay_control(self._agent), 500_000)
        # Side-effect: the underlying buffer's capacity was actually resized.
        self.assertEqual(self._agent.replay_buffer.capacity, 500_000)

    def test_set_mreplay_within_bounds(self):
        # M_replay=15_000 with min=10_000 / max=20_000 falls inside the
        # window, so the function should return the unclamped value.
        self.assertEqual(
            replay_control(
                self._agent,
                M_replay=15_000,
                min_replay_size=10_000,
                max_replay_size=20_000,
            ),
            15_000,
        )
        self.assertEqual(self._agent.replay_buffer.capacity, 15_000)

    def test_clamp_to_max(self):
        self.assertEqual(
            replay_control(
                self._agent,
                M_replay=10_000_000,
                min_replay_size=10_000,
                max_replay_size=20_000,
            ),
            20_000,
        )
        self.assertEqual(self._agent.replay_buffer.capacity, 20_000)


if __name__ == "__main__":
    unittest.main()
