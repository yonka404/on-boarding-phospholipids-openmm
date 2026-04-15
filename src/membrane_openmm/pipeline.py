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
    platform_name: str | None = None,
    stages: Iterable[Stage] = DEFAULT_STAGES,
    report_interval: int = 5_000,
    checkpoint_interval: int = 50_000,
    continue_from_existing: bool = True,
) -> None:

    files = CharmmGuiFiles.from_root(inputs_root=inputs_root)
    system = CharmmGuiSystem(files=files)
    loaded = system.load()

    # TODO: I think we should use here the validated output dir now
    _write_metadata(outdir, loaded, temperature_k)




    # previous_state_xml: Path | None = None
    #
    # for index, stage in enumerate(stages, start=1):
    #     stage_dir = outdir / f"{index:02d}_{stage.name}"
    #     stage_dir.mkdir(parents=True, exist_ok=True)
    #     state_xml = stage_dir / f"{stage.name}.state.xml"
    #     checkpoint_path = stage_dir / f"{stage.name}.chk"
    #     thermo_csv = stage_dir / f"{stage.name}.thermo.csv"
    #     dcd_path = stage_dir / f"{stage.name}.dcd"
    #     final_pdb = stage_dir / f"{stage.name}.final.pdb"
    #
    #     if continue_from_existing and state_xml.exists():
    #         previous_state_xml = state_xml
    #         print(f"[skip] {stage.name} already finished -> {state_xml}")
    #         continue
    #
    #     print(
    #         f"[run] {stage.name}: steps={stage.steps}, dt_fs={stage.dt_fs}, membrane_barostat={stage.use_membrane_barostat}"
    #     )
    #     simulation = _build_simulation(loaded, temperature_k, stage, platform_name)
    #     dof = _compute_degrees_of_freedom(simulation.system)
    #
    #     if previous_state_xml is None:
    #         simulation.context.setPositions(loaded.pdb.positions)
    #         simulation.context.setVelocitiesToTemperature(temperature_k * kelvin)
    #         if stage.minimize_first:
    #             simulation.minimizeEnergy(maxIterations=10_000)
    #     else:
    #         simulation.loadState(str(previous_state_xml))
    #
    #     simulation.reporters.append(DCDReporter(str(dcd_path), report_interval))
    #
    #     for completed in range(0, stage.steps, report_interval):
    #         n = min(report_interval, stage.steps - completed)
    #         simulation.step(n)
    #
    #         state = simulation.context.getState(getEnergy=True)
    #         step_now = completed + n
    #         time_ps = step_now * stage.dt_fs / 1000.0
    #         ke = state.getKineticEnergy().value_in_unit(kilojoule_per_mole)
    #         pe = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    #         temperature = (
    #             2.0 * state.getKineticEnergy() / (dof * MOLAR_GAS_CONSTANT_R)
    #         ).value_in_unit(kelvin)
    #         lx, ly, lz = _box_vectors_to_lengths_nm(state)
    #         volume_nm3 = _volume_nm3(lx, ly, lz)
    #         area_xy_nm2 = lx * ly
    #         area_per_lipid_nm2 = (
    #             area_xy_nm2 / loaded.metadata.nliptop
    #             if loaded.metadata.nliptop
    #             else float("nan")
    #         )
    #
    #         row = {
    #             "stage": stage.name,
    #             "step": step_now,
    #             "time_ps": round(time_ps, 6),
    #             "dt_fs": stage.dt_fs,
    #             "temperature_k": round(float(temperature), 6),
    #             "potential_energy_kj_mol": round(float(pe), 6),
    #             "kinetic_energy_kj_mol": round(float(ke), 6),
    #             "lx_nm": round(lx, 6),
    #             "ly_nm": round(ly, 6),
    #             "lz_nm": round(lz, 6),
    #             "area_xy_nm2": round(area_xy_nm2, 6),
    #             "area_per_lipid_nm2": round(area_per_lipid_nm2, 6),
    #             "volume_nm3": round(volume_nm3, 6),
    #         }
    #         _append_csv_row(thermo_csv, row)
    #
    #         if checkpoint_interval > 0 and step_now % checkpoint_interval == 0:
    #             simulation.saveCheckpoint(str(checkpoint_path))
    #
    #     simulation.saveState(str(state_xml))
    #     _save_final_pdb(simulation, final_pdb)
    #     previous_state_xml = state_xml
