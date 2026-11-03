"""Tiny real OpenMM recovery: no production inputs, hardware or simulation runs."""

import csv
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openmm import CustomExternalForce, Platform, System, Vec3, VerletIntegrator
from openmm.app import Simulation

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.simulation.reporters import StageReporterInstaller
from charmm_gui_md.shared.workflows.stage_runner import StageRunner
from charmm_gui_md.solution.profile import SOLUTION_PROFILE
from tests.simulation.test_recovery import topology


def tiny_simulation():
    system = System()
    system.addParticle(39.9)
    force = CustomExternalForce("0.5*x*x")
    force.addParticle(0, [])
    system.addForce(force)
    simulation = Simulation(
        topology(),
        system,
        VerletIntegrator(0.002),
        Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions([Vec3(0.2, 0.1, 0.3)])
    simulation.context.setVelocities([Vec3(0.1, 0.0, 0.0)])
    return simulation


class RealCheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.outputs = Path(directory.name)
        self.artifacts = StageArtifacts.for_stage(self.outputs, "step5_production")
        self.artifacts.create_directory()
        self.protocol = SimpleNamespace(
            step_name="step5_production",
            dynamics_steps=20,
            timestep_ps=0.002,
            state_report_interval_steps=2,
            trajectory_report_interval_steps=4,
            has_minimization=True,
        )
        self.simulation = tiny_simulation()
        StageReporterInstaller().install(self.simulation, self.artifacts, self.protocol)
        self.simulation.step(8)
        saved_checkpoint = self.artifacts.checkpoint_path.read_bytes()
        self.simulation.step(4)
        self.simulation.reporters.clear()
        # Crash after reporters were written, before the checkpoint was replaced.
        self.artifacts.checkpoint_path.write_bytes(saved_checkpoint)

    def run_recovery(self, simulation):
        files = SimpleNamespace(
            inputs_dir=Path("unused"), initial_coordinates_path=Path("unused")
        )
        with (
            mock.patch(
                "charmm_gui_md.shared.workflows.stage_runner.OpenMMNativeFiles.from_root",
                return_value=files,
            ),
            mock.patch(
                "charmm_gui_md.shared.workflows.stage_runner.OpenMMStageProtocol.from_file",
                return_value=self.protocol,
            ),
            mock.patch(
                "charmm_gui_md.shared.workflows.stage_runner.OpenMMSimulationFactory.create",
                return_value=simulation,
            ),
            mock.patch.object(
                simulation,
                "minimizeEnergy",
                side_effect=AssertionError("must not minimize"),
            ),
            mock.patch.object(
                simulation.context,
                "setVelocitiesToTemperature",
                side_effect=AssertionError("must not generate velocities"),
            ),
        ):
            return StageRunner(SOLUTION_PROFILE).run(
                Path("unused"), self.outputs, "step5_production"
            )

    def test_resume_repairs_then_appends_without_duplicate_steps_or_frames(self):
        resumed = tiny_simulation()
        try:
            self.run_recovery(resumed)
        finally:
            resumed.reporters.clear()
        self.assertEqual(resumed.currentStep, 20)
        with self.artifacts.state_data_path.open() as handle:
            records = list(csv.reader(handle))
        self.assertEqual([int(row[0]) for row in records[1:]], list(range(2, 21, 2)))
        dcd = self.artifacts.trajectory_path.read_bytes()
        self.assertEqual(struct.unpack_from("<i", dcd, 8)[0], 5)
        self.assertEqual(struct.unpack_from("<i", dcd, 20)[0], 20)
        self.assertEqual(len(dcd), 276 + 5 * 92)
        # A continuous deterministic reference reaches the same physical state.
        self.simulation.step(8)
        self.assertEqual(
            resumed.context.getState(getPositions=True).getPositions(),
            self.simulation.context.getState(getPositions=True).getPositions(),
        )
        self.assertTrue(self.artifacts.is_complete)
        self.assertFalse(self.artifacts.checkpoint_path.exists())
        self.assertEqual(len(list(self.artifacts.output_dir.glob("recovery-*"))), 1)

    def test_incompatible_checkpoint_preserves_reporters_and_finals(self):
        self.artifacts.final_state_path.write_text("interrupted finalization")
        self.artifacts.final_coordinates_path.write_text("interrupted finalization")
        self.artifacts.checkpoint_path.write_bytes(b"not an OpenMM checkpoint")
        before = {p.name: p.read_bytes() for p in self.artifacts.output_dir.iterdir()}
        with self.assertRaisesRegex(RuntimeError, "Could not load checkpoint"):
            self.run_recovery(tiny_simulation())
        self.assertEqual(
            before,
            {p.name: p.read_bytes() for p in self.artifacts.output_dir.iterdir()},
        )

    def test_checkpoint_at_stage_target_finishes_interrupted_finalization(self):
        self.protocol.dynamics_steps = 8
        self.artifacts.final_state_path.write_text("partial state")
        self.artifacts.final_coordinates_path.write_text("partial coordinates")
        resumed = tiny_simulation()
        try:
            with mock.patch.object(
                resumed, "step", side_effect=AssertionError("nothing remains")
            ):
                self.run_recovery(resumed)
        finally:
            resumed.reporters.clear()
        self.assertTrue(self.artifacts.is_complete)
        self.assertFalse(self.artifacts.checkpoint_path.exists())
        self.assertEqual(self.artifacts.trajectory_path.stat().st_size, 460)
