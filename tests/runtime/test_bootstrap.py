import os

os.environ["CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP"] = "1"

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from charmm_gui_md.shared import runtime


class RuntimeBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime._BOOTSTRAP_ATTEMPTED = False

    def test_bootstrap_honors_generic_skip_environment_variable(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP": "1"},
                clear=True,
            ),
            mock.patch("charmm_gui_md.shared.runtime.platform.system") as system,
        ):
            runtime.bootstrap_openmm_runtime()

        system.assert_not_called()

    def test_discover_rocm_roots_orders_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            opt_root = Path(tmpdir)
            rocm_root = opt_root / "rocm"
            rocm_root.mkdir()
            old_root = opt_root / "rocm-6.4.0"
            old_root.mkdir()
            new_root = opt_root / "rocm-7.2.2"
            new_root.mkdir()
            hipconfig_root = opt_root / "hipconfig-root"
            (hipconfig_root / "bin").mkdir(parents=True)
            (hipconfig_root / "bin" / "hipconfig").write_text("")

            roots = runtime.discover_rocm_roots(
                environ={
                    "ROCM_PATH": str(new_root),
                    "HIP_PATH": str(old_root),
                },
                which=lambda _: str(hipconfig_root / "bin" / "hipconfig"),
                opt_root=opt_root,
            )

            self.assertEqual(
                roots,
                (
                    new_root.resolve(),
                    old_root.resolve(),
                    hipconfig_root.resolve(),
                    rocm_root.resolve(),
                ),
            )

    def test_bootstrap_tries_next_candidate_after_preload_failure(self) -> None:
        first_root = Path("/rocm-a")
        second_root = Path("/rocm-b")
        first_libs = (
            first_root / "lib/libamdhip64.so.7",
            first_root / "lib/libhiprtc.so.7",
        )
        second_libs = (
            second_root / "lib/libamdhip64.so.7",
            second_root / "lib/libhiprtc.so.7",
        )

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch(
                "charmm_gui_md.shared.runtime.platform.system", return_value="Linux"
            ),
            mock.patch(
                "charmm_gui_md.shared.runtime.discover_rocm_roots",
                return_value=(first_root, second_root),
            ),
            mock.patch(
                "charmm_gui_md.shared.runtime.hip_runtime_libraries",
                side_effect=(first_libs, second_libs),
            ),
            mock.patch(
                "charmm_gui_md.shared.runtime.preload_shared_libraries",
                side_effect=(OSError("broken"), None),
            ) as preload,
            mock.patch("charmm_gui_md.shared.runtime.logger") as logger,
        ):
            runtime.bootstrap_openmm_runtime()

        self.assertEqual(preload.call_count, 2)
        logger.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
