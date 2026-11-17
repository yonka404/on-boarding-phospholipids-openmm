import os
import subprocess
import sys
import unittest
from pathlib import Path


class PackageImportTests(unittest.TestCase):
    def test_import_does_not_probe_optional_gpu_packages(self) -> None:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CHARMM_GUI_MD_")
        }
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import importlib.metadata
from unittest import mock

# Load third-party dependencies before checking the project's package import.
import openmm
import pydantic

with mock.patch.object(
    importlib.metadata,
    "version",
    side_effect=AssertionError("Package import must not probe optional GPU packages"),
):
    import charmm_gui_md
""",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
