import os
import tempfile
import unittest
from pathlib import Path

os.environ["MEMBRANE_OPENMM_SKIP_BOOTSTRAP"] = "1"

from openmm import CustomExternalForce, Platform, System, Vec3, VerletIntegrator
from openmm.app import Element, Simulation, Topology
from openmm.unit import nanometer, picosecond

from protein_membrane_md.artifacts import RestartSource
from protein_membrane_md.protocols import OpenMMStageProtocol
from protein_membrane_md.simulation.initialization import SimulationInitializer


class SimulationInitializerTests(unittest.TestCase):
    def test_restart_state_keeps_new_stage_context_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "final_state.xml"
            previous_simulation = _simulation_with_parameters(
                {"fc_cdih": 1000.0, "fc_ldih": 1000.0},
            )
            previous_simulation.context.setPositions([Vec3(1.0, 2.0, 3.0)] * nanometer)
            previous_simulation.context.setVelocities(
                [Vec3(0.1, 0.2, 0.3)] * nanometer / picosecond,
            )
            previous_simulation.saveState(str(state_path))

            next_simulation = _simulation_with_parameters({"fc_ldih": 100.0})

            SimulationInitializer().initialize(
                next_simulation,
                RestartSource(
                    coordinates_path=Path("unused.pdb"),
                    state_path=state_path,
                    description="restart from previous protocol stage",
                ),
                _protocol(),
            )

        self.assertEqual(next_simulation.context.getParameter("fc_ldih"), 100.0)
        state = next_simulation.context.getState(getPositions=True, getVelocities=True)
        self.assertEqual(
            state.getPositions(asNumpy=True).value_in_unit(nanometer).tolist(),
            [[1.0, 2.0, 3.0]],
        )
        self.assertEqual(
            state.getVelocities(asNumpy=True).value_in_unit(nanometer / picosecond).tolist(),
            [[0.1, 0.2, 0.3]],
        )


def _simulation_with_parameters(parameters: dict[str, float]) -> Simulation:
    system = System()
    system.addParticle(12.0)
    for name, value in parameters.items():
        force = CustomExternalForce(f"{name}*x")
        force.addGlobalParameter(name, value)
        force.addParticle(0, [])
        system.addForce(force)

    return Simulation(
        _single_atom_topology(),
        system,
        VerletIntegrator(0.001 * picosecond),
        Platform.getPlatformByName("Reference"),
    )


def _single_atom_topology() -> Topology:
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("C", Element.getByAtomicNumber(6), residue)
    return topology


def _protocol() -> OpenMMStageProtocol:
    return OpenMMStageProtocol(
        step_name="step6.6_equilibration",
        minimization_steps=None,
        minimization_tolerance_kj_mol_nm=None,
        generate_velocities=False,
        velocity_temperature_kelvin=None,
        dynamics_steps=10,
        timestep_ps=0.001,
        state_report_interval_steps=10,
        trajectory_report_interval_steps=10,
        temperature_kelvin=303.15,
        friction_per_ps=1.0,
        switch_distance_nm=1.0,
        cutoff_distance_nm=1.2,
        ewald_tolerance=0.0005,
        constraints_name="HBonds",
        pressure_coupling=False,
        pressure_bar=None,
        barostat_kind=None,
        membrane_xy_mode_name=None,
        membrane_z_mode_name=None,
        surface_tension_dyne_per_cm=None,
        barostat_interval_steps=None,
    )


if __name__ == "__main__":
    unittest.main()
