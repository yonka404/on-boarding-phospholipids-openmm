"""Recovery must retain exactly the checkpoint prefix before append is opened."""

import csv
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openmm import Vec3
from openmm.app import DCDFile, Topology, element

from charmm_gui_md.shared.artifacts import StageArtifacts
from charmm_gui_md.shared.simulation.reporters import StageReporterInstaller

HEADERS = (
    "Step",
    "Time (ps)",
    "Potential Energy (kJ/mole)",
    "Kinetic Energy (kJ/mole)",
    "Total Energy (kJ/mole)",
    "Temperature (K)",
    "Box Volume (nm^3)",
    "Density (g/mL)",
    "Speed (ns/day)",
)


def topology(periodic=True):
    top = Topology()
    residue = top.addResidue("Ar", top.addChain())
    top.addAtom("Ar", element.argon, residue)
    if periodic:
        top.setPeriodicBoxVectors((Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)))
    return top


def write_reports(artifacts, top, *, steps=(2, 4, 6, 8, 10, 12), frames=3):
    artifacts.create_directory()
    artifacts.checkpoint_path.write_bytes(b"unchanged checkpoint")
    with artifacts.state_data_path.open("w", newline="") as handle:
        handle.write('#"' + '","'.join(HEADERS) + '"\n')
        for step in steps:
            handle.write(f"{step},{step * 0.002},1,2,3,300,8,0.1,--\n")
    with artifacts.trajectory_path.open("wb") as handle:
        dcd = DCDFile(handle, top, 0.002, firstStep=4, interval=4)
        for i in range(frames):
            dcd.writeModel([Vec3(i * 0.01, 0, 0)])


def csv_steps(path):
    with path.open() as handle:
        return [int(row[0]) for row in list(csv.reader(handle))[1:]]


class ReporterRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.artifacts = StageArtifacts.for_stage(
            Path(temporary.name), "step5_production"
        )
        self.topology = topology()
        self.protocol = SimpleNamespace(
            step_name="step5_production",
            dynamics_steps=20,
            timestep_ps=0.002,
            state_report_interval_steps=2,
            trajectory_report_interval_steps=4,
        )
        self.simulation = SimpleNamespace(
            reporters=[], currentStep=8, topology=self.topology
        )
        self.addCleanup(self.simulation.reporters.clear)
        write_reports(self.artifacts, self.topology)

    def install(self):
        StageReporterInstaller().install(
            self.simulation,
            self.artifacts,
            self.protocol,
            append=True,
        )
        self.simulation.reporters.clear()

    def snapshot(self):
        return {
            name: (self.artifacts.output_dir / name).read_bytes()
            for name in ("state_data.csv", "trajectory.dcd", "checkpoint.chk")
        }

    def backups(self):
        return list(self.artifacts.output_dir.glob("recovery-*"))

    def test_overlap_is_trimmed_and_both_originals_are_backed_up(self):
        original = self.snapshot()
        self.install()
        self.assertEqual(csv_steps(self.artifacts.state_data_path), [2, 4, 6, 8])
        dcd = self.artifacts.trajectory_path.read_bytes()
        self.assertEqual(struct.unpack_from("<i", dcd, 8)[0], 2)
        self.assertEqual(struct.unpack_from("<i", dcd, 20)[0], 8)
        self.assertEqual(len(dcd), 276 + 2 * 92)
        self.assertEqual(dcd[276:], original["trajectory.dcd"][276:460])
        self.assertEqual(len(self.backups()), 1)
        for name in ("state_data.csv", "trajectory.dcd"):
            self.assertEqual((self.backups()[0] / name).read_bytes(), original[name])
        self.assertEqual(
            self.artifacts.checkpoint_path.read_bytes(), original["checkpoint.chk"]
        )

    def test_incomplete_tails_are_discarded_if_retained_prefix_is_intact(self):
        self.artifacts.state_data_path.write_bytes(
            self.artifacts.state_data_path.read_bytes() + b"14,partial"
        )
        self.artifacts.trajectory_path.write_bytes(
            self.artifacts.trajectory_path.read_bytes()[:-9]
        )
        self.install()
        self.assertEqual(csv_steps(self.artifacts.state_data_path), [2, 4, 6, 8])
        self.assertEqual(self.artifacts.trajectory_path.stat().st_size, 460)

    def test_repair_is_idempotent_and_valid_prefix_needs_no_new_backup(self):
        self.install()
        repaired = self.snapshot()
        self.install()
        self.assertEqual(self.snapshot(), repaired)
        self.assertEqual(len(self.backups()), 1)

    def test_nonperiodic_dcd_is_supported(self):
        self.simulation.topology = topology(periodic=False)
        write_reports(self.artifacts, self.simulation.topology)
        self.install()
        self.assertEqual(self.artifacts.trajectory_path.stat().st_size, 276 + 2 * 36)

    def test_checkpoint_between_report_intervals_retains_only_due_records(self):
        self.simulation.currentStep = 9
        self.install()
        self.assertEqual(csv_steps(self.artifacts.state_data_path), [2, 4, 6, 8])
        self.assertEqual(self.artifacts.trajectory_path.stat().st_size, 460)

    def test_invalid_retained_csv_preserves_every_file(self):
        valid = self.artifacts.state_data_path.read_bytes()
        for broken in (
            valid.replace(b"4,0.008", b"6,0.008"),
            valid.replace(b"4,0.008", b"4,NaN"),
            valid.replace(b'"Step"', b'"Unknown"'),
            b"\n".join(valid.splitlines()[:4]) + b"\n8,partial",
        ):
            with self.subTest(broken=broken[:40]):
                self.artifacts.state_data_path.write_bytes(broken)
                original = self.snapshot()
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError), "state_data.csv"
                ):
                    self.install()
                self.assertEqual(self.snapshot(), original)
                self.assertEqual(self.simulation.reporters, [])
                self.assertEqual(self.backups(), [])

    def test_invalid_retained_dcd_preserves_every_file(self):
        valid = self.artifacts.trajectory_path.read_bytes()
        cases = [valid[:450]]
        for offset, value in (
            (0, 42),
            (12, 7),
            (16, 8),
            (48, 2),
            (268, 2),
            (332, 7),
            (8, 1),
        ):
            broken = bytearray(valid)
            struct.pack_into("<i", broken, offset, value)
            cases.append(bytes(broken))
        broken = bytearray(valid)
        struct.pack_into("<f", broken, 336, float("nan"))
        cases.append(bytes(broken))
        for broken in cases:
            with self.subTest(header=broken[:24]):
                self.artifacts.trajectory_path.write_bytes(broken)
                original = self.snapshot()
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError), "trajectory.dcd"
                ):
                    self.install()
                self.assertEqual(self.snapshot(), original)
                self.assertEqual(self.simulation.reporters, [])
                self.assertEqual(self.backups(), [])

    def test_backup_failure_changes_neither_original(self):
        original = self.snapshot()
        with (
            mock.patch("shutil.copy2", side_effect=OSError("quota exceeded")),
            self.assertRaisesRegex(OSError, "quota exceeded"),
        ):
            self.install()
        self.assertEqual(self.snapshot(), original)
        self.assertEqual(self.simulation.reporters, [])

    def test_interruption_between_replacements_keeps_backups_and_can_retry(self):
        original = self.snapshot()
        import os

        replace = os.replace

        def interrupt_second(source, target):
            # Both backups must be complete before even the first replacement.
            for name in ("state_data.csv", "trajectory.dcd"):
                self.assertEqual(
                    (self.backups()[0] / name).read_bytes(), original[name]
                )
            if Path(target) == self.artifacts.trajectory_path:
                raise KeyboardInterrupt()
            return replace(source, target)

        with (
            mock.patch("os.replace", side_effect=interrupt_second),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.install()
        self.assertEqual(csv_steps(self.artifacts.state_data_path), [2, 4, 6, 8])
        self.assertEqual(
            self.artifacts.trajectory_path.read_bytes(), original["trajectory.dcd"]
        )
        self.assertEqual(
            self.artifacts.checkpoint_path.read_bytes(), original["checkpoint.chk"]
        )
        self.install()
        self.assertEqual(self.artifacts.trajectory_path.stat().st_size, 460)
        self.assertEqual(len(self.backups()), 2)

    def test_second_backup_failure_does_not_replace_first_reporter(self):
        original = self.snapshot()
        copy = shutil.copy2

        def fail_second(source, target):
            if Path(source) == self.artifacts.trajectory_path:
                raise OSError("backup failed")
            return copy(source, target)

        with (
            mock.patch("shutil.copy2", side_effect=fail_second),
            self.assertRaisesRegex(OSError, "backup failed"),
        ):
            self.install()
        self.assertEqual(self.snapshot(), original)

    def test_zero_step_checkpoint_retains_valid_empty_reporter_headers(self):
        self.simulation.currentStep = 0
        self.install()
        self.assertEqual(csv_steps(self.artifacts.state_data_path), [])
        dcd = self.artifacts.trajectory_path.read_bytes()
        self.assertEqual(len(dcd), 276)
        self.assertEqual(struct.unpack_from("<i", dcd, 8)[0], 0)
        self.assertEqual(struct.unpack_from("<i", dcd, 20)[0], 0)

    def test_dcd_rescaled_for_large_step_numbers_can_append_after_repair(self):
        self.protocol.trajectory_report_interval_steps = 1_000_000_000
        self.protocol.state_report_interval_steps = 1_000_000_000
        self.protocol.dynamics_steps = 4_000_000_000
        self.simulation.currentStep = 2_000_000_000
        self.artifacts.state_data_path.write_text(
            '#"' + '","'.join(HEADERS) + '"\n'
            "1000000000,2000000,1,2,3,300,8,0.1,--\n"
            "2000000000,4000000,1,2,3,300,8,0.1,--\n"
            "3000000000,6000000,1,2,3,300,8,0.1,--\n"
        )
        with self.artifacts.trajectory_path.open("wb") as handle:
            dcd = DCDFile(
                handle,
                self.topology,
                0.002,
                firstStep=1_000_000_000,
                interval=1_000_000_000,
            )
            for _ in range(3):
                dcd.writeModel([Vec3(0, 0, 0)])
        self.install()
        with self.artifacts.trajectory_path.open("r+b") as handle:
            dcd = DCDFile(
                handle,
                self.topology,
                0.002,
                firstStep=1_000_000_000,
                interval=1_000_000_000,
                append=True,
            )
            dcd.writeModel([Vec3(0, 0, 0)])
        data = self.artifacts.trajectory_path.read_bytes()
        self.assertEqual(struct.unpack_from("<4i", data, 8), (3, 1, 1, 3))
        self.assertEqual(len(data), 276 + 3 * 92)

    def test_scaled_dcd_rolled_back_before_overflow_keeps_consistent_timing(self):
        self.protocol.trajectory_report_interval_steps = 400_000_000
        self.protocol.state_report_interval_steps = 800_000_000
        self.protocol.dynamics_steps = 4_000_000_000
        self.simulation.currentStep = 800_000_000
        self.artifacts.state_data_path.write_text(
            '#"' + '","'.join(HEADERS) + '"\n800000000,1600000,1,2,3,300,8,0.1,--\n'
        )
        with self.artifacts.trajectory_path.open("wb") as handle:
            dcd = DCDFile(
                handle,
                self.topology,
                0.002,
                firstStep=400_000_000,
                interval=400_000_000,
            )
            for _ in range(6):
                dcd.writeModel([Vec3(0, 0, 0)])
        self.install()
        with self.artifacts.trajectory_path.open("r+b") as handle:
            dcd = DCDFile(
                handle,
                self.topology,
                0.002,
                firstStep=400_000_000,
                interval=400_000_000,
                append=True,
            )
            dcd.writeModel([Vec3(0, 0, 0)])
        data = self.artifacts.trajectory_path.read_bytes()
        self.assertEqual(
            struct.unpack_from("<4i", data, 8),
            (3, 400_000_000, 400_000_000, 1_200_000_000),
        )
        self.assertAlmostEqual(
            struct.unpack_from("<f", data, 44)[0], 0.002 / 0.04888821
        )
        self.assertEqual(len(data), 276 + 3 * 92)
