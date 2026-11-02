"""Submit a full solution protocol to SIMCALC from its login node."""

import sys
from pathlib import Path

# When this file is executed directly (for example, with
# `python3 mains/simcalc.py`), Python does not automatically include the
# repository root in its import path. Add that root directory so local
# packages such as `mains` and `charmm_gui_md` can be imported. This is
# unnecessary when the file is run as a module with `python -m mains.simcalc`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mains._simcalc.submit import main

if __name__ == "__main__":
    raise SystemExit(main())
