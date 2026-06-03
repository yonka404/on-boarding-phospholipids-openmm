import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP"] = "1"

from openmm import (
    CustomBondForce,
    CustomExternalForce,
    CustomNonbondedForce,
    CustomTorsionForce,
    MonteCarloBarostat,
    MonteCarloMembraneBarostat,
    NonbondedForce,
    System,
    Vec3,
)
from openmm.unit import angstrom, kilojoule_per_mole, nanometer

from charmm_gui_md.shared.protocols import OpenMMStageProtocol
from charmm_gui_md.shared.simulation import OpenMMSimulationFactory


class OpenmmNativeSystemDetailTests(unittest.TestCase):
    def test_pressure_protocols_select_environment_appropriate_barostats(self) -> None:
        factory = OpenMMSimulationFactory()

        isotropic = factory._build_barostat(
            _protocol(
                pressure_coupling=True,
                pressure_bar=1.0,
                barostat_kind="isotropic",
                barostat_interval_steps=25,
            )
        )
        membrane = factory._build_barostat(
            _protocol(
                pressure_coupling=True,
                pressure_bar=1.0,
                barostat_kind="membrane",
                membrane_xy_mode_name="XYIsotropic",
                membrane_z_mode_name="ZFree",
                surface_tension_dyne_per_cm=0.0,
                barostat_interval_steps=25,
            )
        )

        self.assertIsInstance(isotropic, MonteCarloBarostat)
        self.assertIsInstance(membrane, MonteCarloMembraneBarostat)

    def test_create_applies_charmm_gui_force_switch_vdw(self) -> None:
        factory = OpenMMSimulationFactory()
        psf = _FakePsf(_system_with_nonbonded_force())
        files = _FakeFiles(psf=psf)
        protocol = _protocol(restraints_enabled=False, vdw_method_name="Force-switch")
        captured: dict[str, System] = {}

        with mock.patch.object(
            factory,
            "_create_simulation",
            side_effect=lambda topology, system, integrator, step_name: captured.setdefault(
                "system",
                system,
            ),
        ):
            factory.create(files, protocol)

        self.assertNotIn("switchDistance", psf.create_system_kwargs)

        system = captured["system"]
        custom_nonbonded = _single_force(system, CustomNonbondedForce)
        custom_bond = _single_force(system, CustomBondForce)
        self.assertEqual(custom_nonbonded.getNumParticles(), 2)
        self.assertEqual(custom_bond.getNumBonds(), 1)

        nonbonded = _single_force(system, NonbondedForce)
        _, sigma, epsilon = nonbonded.getParticleParameters(0)
        self.assertEqual(sigma.value_in_unit(nanometer), 0.0)
        self.assertEqual(epsilon.value_in_unit(kilojoule_per_mole), 0.0)

        _, _, _, sigma14, epsilon14 = nonbonded.getExceptionParameters(0)
        self.assertEqual(sigma14.value_in_unit(nanometer), 0.0)
        self.assertEqual(epsilon14.value_in_unit(kilojoule_per_mole), 0.0)

    def test_create_applies_enabled_openmm_native_restraints(self) -> None:
        factory = OpenMMSimulationFactory()
        psf = _FakePsf(_system_with_particles(5))

        with tempfile.TemporaryDirectory() as tmpdir:
            inputs_dir = Path(tmpdir)
            restraints_dir = inputs_dir / "restraints"
            restraints_dir.mkdir()
            (restraints_dir / "prot_pos.txt").write_text("0 BB\n1 SC\n")
            (restraints_dir / "lipid_pos.txt").write_text("2\n")
            (restraints_dir / "dihe.txt").write_text("0 1 2 3 60.0 20.0\n")
            (restraints_dir / "carbohydrate_restraint.dat").write_text(
                "1 2 3 4 180.0 30.0\n"
            )

            files = _FakeFiles(psf=psf, inputs_dir=inputs_dir)
            protocol = _protocol(
                restraints_enabled=True,
                vdw_method_name="Switch",
                protein_backbone_restraint_kj_mol_nm2=4000.0,
                protein_side_chain_restraint_kj_mol_nm2=2000.0,
                lipid_position_restraint_kj_mol_nm2=1000.0,
                lipid_dihedral_restraint_kj_mol_rad2=1000.0,
                carbohydrate_dihedral_restraint_kj_mol_rad2=1000.0,
            )
            captured: dict[str, System] = {}

            with mock.patch.object(
                factory,
                "_create_simulation",
                side_effect=lambda topology, system, integrator, step_name: captured.setdefault(
                    "system",
                    system,
                ),
            ):
                factory.create(files, protocol)

        system = captured["system"]
        external_forces = _forces(system, CustomExternalForce)
        torsion_forces = _forces(system, CustomTorsionForce)

        self.assertEqual([force.getNumParticles() for force in external_forces], [2, 1])
        self.assertEqual([force.getNumTorsions() for force in torsion_forces], [1, 1])


class _FakeFiles:
    def __init__(
        self,
        *,
        psf: "_FakePsf",
        inputs_dir: Path = Path("."),
    ) -> None:
        self.inputs_dir = inputs_dir
        self._psf = psf
        self._pdb = _FakePdbFile()

    @property
    def psf_file(self) -> "_FakePsf":
        return self._psf

    @property
    def pdb_file(self) -> "_FakePdbFile":
        return self._pdb

    @property
    def params_file(self):
        return object()

    @property
    def box_lengths_angstrom(self) -> tuple[float, float, float]:
        return 10.0, 10.0, 10.0


class _FakePsf:
    NONBONDED_FORCE_GROUP = 1
    topology = object()

    def __init__(self, system: System) -> None:
        self._system = system
        self.create_system_kwargs = {}

    def setBox(self, a_length, b_length, c_length) -> None:
        self.box = a_length, b_length, c_length

    def createSystem(self, params, **kwargs) -> System:
        self.create_system_kwargs = kwargs
        return self._system


class _FakePdbFile:
    positions = [
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.1, 0.2, 0.3),
        Vec3(0.2, 0.3, 0.4),
        Vec3(0.3, 0.4, 0.5),
        Vec3(0.4, 0.5, 0.6),
    ] * nanometer


def _protocol(**overrides) -> OpenMMStageProtocol:
    values = {
        "step_name": "step6.1_equilibration",
        "minimization_steps": None,
        "minimization_tolerance_kj_mol_nm": None,
        "generate_velocities": False,
        "velocity_temperature_kelvin": None,
        "dynamics_steps": 10,
        "timestep_ps": 0.001,
        "state_report_interval_steps": 10,
        "trajectory_report_interval_steps": 10,
        "temperature_kelvin": 303.15,
        "friction_per_ps": 1.0,
        "switch_distance_nm": 1.0,
        "cutoff_distance_nm": 1.2,
        "ewald_tolerance": 0.0005,
        "constraints_name": "HBonds",
        "pressure_coupling": False,
        "pressure_bar": None,
        "barostat_kind": None,
        "membrane_xy_mode_name": None,
        "membrane_z_mode_name": None,
        "surface_tension_dyne_per_cm": None,
        "barostat_interval_steps": None,
    }
    values.update(overrides)
    return OpenMMStageProtocol(**values)


def _system_with_particles(count: int) -> System:
    system = System()
    for _ in range(count):
        system.addParticle(12.0)
    return system


def _system_with_nonbonded_force() -> System:
    system = _system_with_particles(2)
    force = NonbondedForce()
    force.addParticle(0.1, 3.0 * angstrom, 0.2 * kilojoule_per_mole)
    force.addParticle(-0.1, 3.2 * angstrom, 0.3 * kilojoule_per_mole)
    force.addException(
        0,
        1,
        0.0,
        2.5 * angstrom,
        0.1 * kilojoule_per_mole,
    )
    system.addForce(force)
    return system


def _forces(system: System, force_type):
    return [force for force in system.getForces() if isinstance(force, force_type)]


def _single_force(system: System, force_type):
    forces = _forces(system, force_type)
    if len(forces) != 1:
        raise AssertionError(f"Expected one {force_type.__name__}, found {len(forces)}")
    return forces[0]


if __name__ == "__main__":
    unittest.main()
