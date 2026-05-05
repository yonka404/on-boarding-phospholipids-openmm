from pathlib import Path
from typing import ClassVar

from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile
from pydantic import BaseModel, field_validator


class OpenmmNativeFiles(BaseModel):
    """Validated CHARMM-GUI files needed for the OpenMM setup."""

    model_config = {"frozen": True}

    REQUIRED_DIRECTORIES: ClassVar[tuple[str, ...]] = (
        "lig",
        "toppar",
    )

    REQUIRED_FILES: ClassVar[tuple[str, ...]] = (
        "membrane_restraint.charmm_openmm.str",
        "step5_assembly.psf",
        "step5_assembly.pdb",
        "step5_assembly.str",
        "step6.1_equilibration.inp",
        "step6.2_equilibration.inp",
        "step6.3_equilibration.inp",
        "step6.4_equilibration.inp",
        "step6.5_equilibration.inp",
        "step6.6_equilibration.inp",
        "step7_production.inp",
        "toppar.str",
