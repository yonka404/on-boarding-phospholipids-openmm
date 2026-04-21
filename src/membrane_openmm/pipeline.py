from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from openmm import Platform
from openmm.app import DCDReporter
from openmm.unit import (
    MOLAR_GAS_CONSTANT_R,
    kelvin,
    kilojoule_per_mole,
    nanometer,
)

from membrane_openmm.charmm_gui import CharmmGuiFiles, CharmmGuiSystem


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    dt_fs: float
    use_membrane_barostat: bool
    minimize_first: bool = False


DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage(
        name="eq1",
        steps=125_000,
        dt_fs=1.0,
        use_membrane_barostat=False,
        minimize_first=True,
    ),
    Stage(name="eq2", steps=125_000, dt_fs=1.0, use_membrane_barostat=False),
    Stage(name="eq3", steps=125_000, dt_fs=1.0, use_membrane_barostat=True),
    Stage(name="eq4", steps=250_000, dt_fs=2.0, use_membrane_barostat=True),
    Stage(name="eq5", steps=250_000, dt_fs=2.0, use_membrane_barostat=True),
    Stage(name="eq6", steps=250_000, dt_fs=2.0, use_membrane_barostat=True),
    Stage(name="prod", steps=500_000, dt_fs=2.0, use_membrane_barostat=True),
)


def _pick_platform(name: str | None) -> Platform | None:
    if not name:
        return None
    return Platform.getPlatformByName(name)


def _compute_degrees_of_freedom(system) -> int:
    dof = 0
    for idx in range(system.getNumParticles()):
        if (
            system.getParticleMass(idx).value_in_unit_system(
                system.getParticleMass(idx).unit.system
            )
            != 0
        ):
            dof += 3
    dof -= system.getNumConstraints()
    # subtract COM remover if present
    for force_index in range(system.getNumForces()):
        force_name = system.getForce(force_index).__class__.__name__
        if force_name == "CMMotionRemover":
            dof -= 3
            break
    return max(dof, 1)


def _box_vectors_to_lengths_nm(state) -> tuple[float, float, float]:
    a, b, c = state.getPeriodicBoxVectors()
    a_nm = np.asarray(a.value_in_unit(nanometer), dtype=float)
    b_nm = np.asarray(b.value_in_unit(nanometer), dtype=float)
    c_nm = np.asarray(c.value_in_unit(nanometer), dtype=float)
    lx = float(np.linalg.norm(a_nm))
    ly = float(np.linalg.norm(b_nm))
    lz = float(np.linalg.norm(c_nm))
    return lx, ly, lz


def _volume_nm3(lx: float, ly: float, lz: float) -> float:
    return lx * ly * lz


#
# def _build_simulation(
#     loaded: LoadedCharmmGuiSystem,
#     temperature_k: float,
#     stage: Stage,
#     platform_name: str | None,
# ) -> Simulation:
#     system = loaded.psf.createSystem(
#         loaded.params,
#         nonbondedMethod=PME,
#         nonbondedCutoff=1.2 * nanometer,
#         switchDistance=1.0 * nanometer,
#         constraints=HBonds,
#     )
#
#     if stage.use_membrane_barostat:
#         system.addForce(
#             MonteCarloMembraneBarostat(
#                 1.0 * bar,
#                 0.0 * bar * nanometer,
#                 MonteCarloMembraneBarostat.XYIsotropic,
#                 MonteCarloMembraneBarostat.ZFree,
#                 temperature_k * kelvin,
#             )
#         )
#
#     integrator = LangevinMiddleIntegrator(
#         temperature_k * kelvin,
#         1.0 / picosecond,
#         stage.dt_fs * femtoseconds,
#     )
#
#     platform = _pick_platform(platform_name)
#     if platform is None:
#         sim = Simulation(loaded.psf.topology, system, integrator)
#     else:
#         sim = Simulation(loaded.psf.topology, system, integrator, platform)
#     return sim
#
#
# def _write_metadata(
#     outdir: Path, loaded: LoadedCharmmGuiSystem, temperature_k: float
# ) -> None:
#     payload = {
#         "system_root": str(loaded.system_root),
#         "temperature_k": temperature_k,
#         "metadata": asdict(loaded.metadata),
#     }
#     (outdir / "metadata.json").write_text(
#         json.dumps(payload, indent=2), encoding="utf-8"
#     )
#
#
# def _append_csv_row(csv_path: Path, row: dict[str, float | int | str]) -> None:
#     write_header = not csv_path.exists()
#     with csv_path.open("a", newline="", encoding="utf-8") as fh:
#         writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
#         if write_header:
#             writer.writeheader()
#         writer.writerow(row)
#
#
# def _save_final_pdb(simulation: Simulation, outpath: Path) -> None:
#     state = simulation.context.getState(getPositions=True)
#     with outpath.open("w", encoding="utf-8") as fh:
#         PDBFile.writeFile(simulation.topology, state.getPositions(), fh)


def run_case(
    inputs_root: Path,
    outdir: str | Path,
    temperature_k: float = 303.15,
) -> None:

    files = CharmmGuiFiles.from_root(inputs_root=inputs_root)
    system = CharmmGuiSystem.from_files(files=files)
    loaded = system.load()

    # TODO: I think we should use here the validated output dir now
    _write_metadata(outdir, loaded, temperature_k)
