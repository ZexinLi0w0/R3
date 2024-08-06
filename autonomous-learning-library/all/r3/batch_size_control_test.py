import unittest

from batch_size_control import BatchControl

class BatchSizeControlTest(unittest.TestCase):

    def setUp(self):
        self._batch_control = BatchControl(min_batch_size=64)

    def test_batch_size_control_episode_sanity_check(self):
        assert(self._batch_control.batch_size_control_episode(M_base=0.0) == 64)
        assert(self._batch_control.batch_size_control_episode(d_base=0.0) == 64)

    def test_batch_size_control_episode_default_values(self):
        assert(self._batch_control.batch_size_control_episode() == 64)

    def test_batch_size_control_step_sanity_check(self):
        assert(self._batch_control.batch_size_control_step(M_base=0.0) == 64)

    def test_batch_size_control_step_default_values(self):
        assert(self._batch_control.batch_size_control_step() == 64)

if __name__ == "__main__":
    unittest.main()