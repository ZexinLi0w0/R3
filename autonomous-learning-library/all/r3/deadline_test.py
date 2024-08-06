import unittest

from deadline import Deadline

class DeadlineTest(unittest.TestCase):

    def test_deadline_first_episode(self):
        """
        Test deadlines for first episode no parameters set.
        """
        test_d = Deadline()
        assert(test_d.calculate_episode_deadline() == 0)
        assert(test_d.deadline_gamma() == 1)

    def test_deadline_set_params(self):
        """
        Test deadlines for first episode parameters set.
        """
        test_d = Deadline(lambda_param=1.0, D=1.0)
        assert(test_d.calculate_episode_deadline() == 1.0)
        assert(test_d.deadline_gamma() == 1)

    def test_deadline_reset(self):
        test_d = Deadline(lambda_param=1.0, D=1.0)
        assert(test_d.calculate_episode_deadline() == 1.0)
        assert(test_d.deadline_gamma() == 1)

        test_d.reset()
        assert(test_d.calculate_episode_deadline() == 0)
        assert(test_d.deadline_gamma() == 1)

if __name__ == "__main__":
    unittest.main()