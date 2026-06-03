import os

if os.environ.get("CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP") != "1":
    from charmm_gui_md.shared.runtime import bootstrap_openmm_runtime

    bootstrap_openmm_runtime()

from charmm_gui_md import membrane, solution

__all__ = ["membrane", "solution"]
