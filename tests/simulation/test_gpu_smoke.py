"""Opt-in five-step GPU check: OPENMM_GPU_SMOKE=1 OPENMM_PLATFORM=OpenCL (or CUDA)."""

import locale
import math
import os
import unittest

from openmm import LangevinMiddleIntegrator, NonbondedForce, System, Vec3, unit
from openmm.app import Topology, element

from charmm_gui_md.shared.simulation import OpenMMSimulationFactory


@unittest.skipUnless(os.environ.get("OPENMM_GPU_SMOKE") == "1", "opt-in GPU smoke test")
class GPUSmokeTests(unittest.TestCase):
    def test_requested_gpu_runs_periodic_pme(self) -> None:
        # ROCm can reset the process locale during initialization. Later tests read UTF-8 logs.
        self.addCleanup(locale.setlocale, locale.LC_ALL, locale.setlocale(locale.LC_ALL))
        requested = os.environ.get("OPENMM_PLATFORM")
        self.assertIn(requested, ("OpenCL", "CUDA"), "Set OPENMM_PLATFORM to a GPU backend")
        topology = Topology()
        residue = topology.addResidue("ION", topology.addChain())
        system = System()
        force = NonbondedForce()
        force.setNonbondedMethod(NonbondedForce.PME)
        force.setCutoffDistance(1 * unit.nanometer)
        force.setEwaldErrorTolerance(1e-4)
        for charge in (1.0, -1.0):
            topology.addAtom("Ar", element.argon, residue)
            system.addParticle(40 * unit.dalton)
            force.addParticle(charge, 0.3, 0.1)
        system.addForce(force)
        box = (Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3)) * unit.nanometer
        topology.setPeriodicBoxVectors(box)
        system.setDefaultPeriodicBoxVectors(*box)
        integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picosecond)
        simulation = OpenMMSimulationFactory()._create_simulation(
            topology, system, integrator, "gpu_smoke",
        )
        simulation.context.setPositions([Vec3(1, 1, 1), Vec3(1.5, 1, 1)] * unit.nanometer)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 123)
        simulation.step(5)
        self.assertEqual(simulation.context.getPlatform().getName(), requested)
        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        self.assertTrue(math.isfinite(energy.value_in_unit(unit.kilojoule_per_mole)))
