import inspect
import unittest

from protein_membrane_md.artifacts import RestartResolver
from protein_membrane_md.workflows import StageRunner, SweepRunner


class RunnerApiTests(unittest.TestCase):
    def test_stage_runner_does_not_expose_protocol_schedule(self) -> None:
        parameters = inspect.signature(StageRunner).parameters

        self.assertNotIn("protocol_schedule", parameters)

    def test_sweep_runner_does_not_expose_protocol_schedule(self) -> None:
        parameters = inspect.signature(SweepRunner).parameters

        self.assertNotIn("protocol_schedule", parameters)

    def test_restart_resolver_uses_default_schedule(self) -> None:
        resolver = RestartResolver()

        self.assertIsInstance(resolver, RestartResolver)


if __name__ == "__main__":
    unittest.main()
