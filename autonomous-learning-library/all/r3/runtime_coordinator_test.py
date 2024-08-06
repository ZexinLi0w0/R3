import unittest

from runtime_coordinator import RuntimeCoordinator

class RuntimeCoordinatorTest(unittest.TestCase):

    def test_default(self):
        coordinator = RuntimeCoordinator()
        assert(coordinator.coordinate() == (1024 / 11024, 10000 / 11024))

    def test_set_memory(self):
        coordinator = RuntimeCoordinator(total_memory=0)
        assert(coordinator.coordinate() == (0, 0))

if __name__ == "__main__":
    unittest.main()