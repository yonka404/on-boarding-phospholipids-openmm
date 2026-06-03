import os
import unittest

os.environ["CHARMM_GUI_MD_OPENMM_SKIP_BOOTSTRAP"] = "1"


class PackageApiTests(unittest.TestCase):
    def test_root_package_exposes_environment_modules(self) -> None:
        import charmm_gui_md
        from charmm_gui_md import membrane, solution

        self.assertIs(charmm_gui_md.membrane, membrane)
        self.assertIs(charmm_gui_md.solution, solution)
        self.assertFalse(hasattr(charmm_gui_md, "MEMBRANE_PROFILE"))
        self.assertFalse(hasattr(charmm_gui_md, "SOLUTION_PROFILE"))

    def test_environment_modules_expose_their_protocol_stages(self) -> None:
        from charmm_gui_md import membrane, solution

        self.assertEqual(
            membrane.PROTOCOL_STAGES,
            (
                "step6.1_equilibration",
                "step6.2_equilibration",
                "step6.3_equilibration",
                "step6.4_equilibration",
                "step6.5_equilibration",
                "step6.6_equilibration",
                "step7_production",
            ),
        )
        self.assertEqual(
            solution.PROTOCOL_STAGES,
            ("step4_equilibration", "step5_production"),
        )


if __name__ == "__main__":
    unittest.main()
