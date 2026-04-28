import os

if os.environ.get("MEMBRANE_OPENMM_SKIP_BOOTSTRAP") != "1":
    from ._runtime import bootstrap_openmm_runtime

    bootstrap_openmm_runtime()
