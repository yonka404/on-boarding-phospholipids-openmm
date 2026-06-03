from pathlib import Path

from openmm import (
    CustomBondForce,
    CustomExternalForce,
    CustomNonbondedForce,
    CustomTorsionForce,
    NonbondedForce,
)
from openmm.unit import nanometer

from charmm_gui_md.shared.protocols import OpenMMStageProtocol


def apply_charmm_gui_force_switch(system, psf, protocol: OpenMMStageProtocol):
    if protocol.vdw_method_name.strip().lower() != "force-switch":
        return system

    nonbonded = _nonbonded_force(system)
    nbfix = _nbfix_force(system)
    r_on = protocol.switch_distance_nm
    r_off = protocol.cutoff_distance_nm

    if nbfix is None:
        vdw_force = CustomNonbondedForce(
            "select(step(r-Ron),(cr12*rjunk12 - cr6*rjunk6),"
            "(ccnba/r^12-ccnbb/r^6-ccnba/onoff^6+ccnbb/onoff^3));"
            "cr12 = ccnba*ofdif6;"
            "cr6 = ccnbb*ofdif3;"
            "rjunk12 = (1.0/r^6-1.0/Roff6)^2;"
            "rjunk6 = (1.0/r^3-1.0/Roff3)^2;"
            "ccnba = 4.0*epsilon*sigma^12;"
            "ccnbb = 4.0*epsilon*sigma^6;"
            "sigma = sigma1+sigma2;"
            "epsilon = epsilon1*epsilon2;"
            "ofdif6 = Roff6/(Roff6 - Ron^6);"
            "ofdif3 = Roff3/(Roff3 - Ron^3);"
            "onoff = Roff * Ron;"
            "Roff6 = Roff^6;"
            "Roff3 = Roff^3;"
            f"Ron = {r_on:f};"
            f"Roff = {r_off:f};"
        )
        vdw_force.addPerParticleParameter("sigma")
        vdw_force.addPerParticleParameter("epsilon")
        vdw_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        vdw_force.setCutoffDistance(nonbonded.getCutoffDistance())

        for particle_index in range(nonbonded.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(particle_index)
            nonbonded.setParticleParameters(particle_index, charge, 0.0, 0.0)
            vdw_force.addParticle([sigma * 0.5, epsilon**0.5])

        for exception_index in range(nonbonded.getNumExceptions()):
            atom1, atom2 = nonbonded.getExceptionParameters(exception_index)[:2]
            vdw_force.addExclusion(atom1, atom2)

        if hasattr(psf, "NONBONDED_FORCE_GROUP"):
            vdw_force.setForceGroup(psf.NONBONDED_FORCE_GROUP)
        system.addForce(vdw_force)
    else:
        nbfix.setEnergyFunction(
            "select(step(r-Ron),(cr12*rjunk12 - cr6*rjunk6),"
            "(ccnba/r^12-ccnbb/r^6-ccnba/onoff^6+ccnbb/onoff^3));"
            "cr12 = ccnba*ofdif6;"
            "cr6  = ccnbb*ofdif3;"
            "rjunk12 = (1.0/r^6-1.0/Roff6)^2;"
            "rjunk6 = (1.0/r^3-1.0/Roff3)^2;"
            "ccnbb = bcoef(type1, type2);"
            "ccnba = acoef(type1, type2)^2;"
            "ofdif6 = Roff6/(Roff6 - Ron^6);"
            "ofdif3 = Roff3/(Roff3 - Ron^3);"
            "onoff = Roff * Ron;"
            "Roff6 = Roff^6;"
            "Roff3 = Roff^3;"
            f"Ron  = {r_on:f};"
            f"Roff = {r_off:f};"
        )
        nbfix.setUseLongRangeCorrection(False)

    vdw_14_force = CustomBondForce(
        "select(step(r-Ron),(cr12*rjunk12 - cr6*rjunk6),"
        "(ccnba/r^12-ccnbb/r^6-ccnba/onoff^6+ccnbb/onoff^3));"
        "cr12 = ccnba*ofdif6;"
        "cr6 = ccnbb*ofdif3;"
        "rjunk12 = (1.0/r^6-1.0/Roff6)^2;"
        "rjunk6 = (1.0/r^3-1.0/Roff3)^2;"
        "ccnba = 4.0*epsilon*sigma^12;"
        "ccnbb = 4.0*epsilon*sigma^6;"
        "ofdif6 = Roff6/(Roff6 - Ron^6);"
        "ofdif3 = Roff3/(Roff3 - Ron^3);"
        "onoff = Roff * Ron;"
        "Roff6 = Roff^6;"
        "Roff3 = Roff^3;"
        f"Ron = {r_on:f};"
        f"Roff = {r_off:f};"
    )
    vdw_14_force.addPerBondParameter("sigma")
    vdw_14_force.addPerBondParameter("epsilon")

    for exception_index in range(nonbonded.getNumExceptions()):
        atom1, atom2, charge, sigma, epsilon = nonbonded.getExceptionParameters(
            exception_index
        )
        nonbonded.setExceptionParameters(exception_index, atom1, atom2, charge, 0.0, 0.0)
        vdw_14_force.addBond(atom1, atom2, [sigma, epsilon])

    system.addForce(vdw_14_force)
    return system


def apply_openmm_native_restraints(
    system,
    protocol: OpenMMStageProtocol,
    inputs_dir: Path,
    reference_positions,
):
    if not protocol.restraints_enabled:
        return system

    restraints_dir = inputs_dir / "restraints"
    _add_protein_position_restraints(
        system,
        protocol,
        restraints_dir / "prot_pos.txt",
        reference_positions,
    )
    _add_lipid_position_restraints(
        system,
        protocol,
        restraints_dir / "lipid_pos.txt",
        reference_positions,
    )
    _add_lipid_dihedral_restraints(system, protocol, restraints_dir / "dihe.txt")
    _add_carbohydrate_dihedral_restraints(system, protocol, restraints_dir)
    return system


def _add_protein_position_restraints(
    system,
    protocol: OpenMMStageProtocol,
    restraint_file: Path,
    positions,
) -> None:
    if (
        protocol.protein_backbone_restraint_kj_mol_nm2 <= 0
        and protocol.protein_side_chain_restraint_kj_mol_nm2 <= 0
    ):
        return

    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2;")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for line in _required_restraint_lines(restraint_file):
        atom_index_text, atom_class = line.split()[:2]
        atom_index = int(atom_index_text)
        force_constant = None
        if atom_class == "BB" and protocol.protein_backbone_restraint_kj_mol_nm2 > 0:
            force_constant = protocol.protein_backbone_restraint_kj_mol_nm2
        elif atom_class == "SC" and protocol.protein_side_chain_restraint_kj_mol_nm2 > 0:
            force_constant = protocol.protein_side_chain_restraint_kj_mol_nm2

        if force_constant is None:
            continue

        x_pos, y_pos, z_pos = positions[atom_index].value_in_unit(nanometer)
        force.addParticle(atom_index, [force_constant, x_pos, y_pos, z_pos])

    system.addForce(force)


def _add_lipid_position_restraints(
    system,
    protocol: OpenMMStageProtocol,
    restraint_file: Path,
    positions,
) -> None:
    if protocol.lipid_position_restraint_kj_mol_nm2 <= 0:
        return

    force = CustomExternalForce("k*periodicdistance(0, 0, z, 0, 0, z0)^2;")
    force.addGlobalParameter("k", protocol.lipid_position_restraint_kj_mol_nm2)
    force.addPerParticleParameter("z0")

    for line in _required_restraint_lines(restraint_file):
        atom_index = int(line.split()[0])
        z_pos = positions[atom_index].value_in_unit(nanometer)[2]
        force.addParticle(atom_index, [z_pos])

    system.addForce(force)


def _add_lipid_dihedral_restraints(
    system,
    protocol: OpenMMStageProtocol,
    restraint_file: Path,
) -> None:
    if protocol.lipid_dihedral_restraint_kj_mol_rad2 <= 0:
        return

    force = _dihedral_restraint_force(
        "fc_ldih",
        protocol.lipid_dihedral_restraint_kj_mol_rad2,
    )
    for line in _required_restraint_lines(restraint_file):
        atom1, atom2, atom3, atom4, theta0, width = line.split()[:6]
        force.addTorsion(
            int(atom1),
            int(atom2),
            int(atom3),
            int(atom4),
            [float(width), float(theta0)],
        )

    system.addForce(force)


def _add_carbohydrate_dihedral_restraints(
    system,
    protocol: OpenMMStageProtocol,
    restraints_dir: Path,
) -> None:
    if protocol.carbohydrate_dihedral_restraint_kj_mol_rad2 <= 0:
        return

    force = _dihedral_restraint_force(
        "fc_cdih",
        protocol.carbohydrate_dihedral_restraint_kj_mol_rad2,
    )
    for restraint_file in (
        restraints_dir / "carbohydrate_restraint.dat",
        restraints_dir / "detergent_carbohydrate_restraint.dat",
    ):
        if not restraint_file.is_file():
            continue
        for line in _restraint_lines(restraint_file):
            atom1, atom2, atom3, atom4, theta0, width = line.split()[:6]
            force.addTorsion(
                int(atom1),
                int(atom2),
                int(atom3),
                int(atom4),
                [float(width), float(theta0)],
            )

    system.addForce(force)


def _dihedral_restraint_force(parameter_name: str, force_constant: float):
    force = CustomTorsionForce(
        f"{parameter_name}*max(0, abs(diff+wrap) - rwidth)^2;"
        "wrap = 2*pi*(step(-diff-pi)-step(diff-pi));"
        "diff = theta - rtheta0;"
        "rtheta0 = theta0*pi/180;"
        "rwidth = width*pi/180;"
    )
    force.addGlobalParameter(parameter_name, force_constant)
    force.addGlobalParameter("pi", 3.141592653589793)
    force.addPerTorsionParameter("width")
    force.addPerTorsionParameter("theta0")
    return force


def _nonbonded_force(system) -> NonbondedForce:
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            return force
    raise ValueError("Force-switch requested but the system has no NonbondedForce")


def _nbfix_force(system) -> CustomNonbondedForce | None:
    for force in system.getForces():
        if (
            isinstance(force, CustomNonbondedForce)
            and force.getNumTabulatedFunctions() == 2
        ):
            return force
    return None


def _required_restraint_lines(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Required restraint file not found: {path}")
    return _restraint_lines(path)


def _restraint_lines(path: Path) -> tuple[str, ...]:
    lines: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", maxsplit=1)[0].split("!", maxsplit=1)[0].strip()
        if line:
            lines.append(line)
    return tuple(lines)
