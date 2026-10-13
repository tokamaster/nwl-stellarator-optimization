"""Check that figure inputs stay together when selecting evaluated cases."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("desc"), "DESC is required for figure-input tests")
class PlotInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plotters = {}
        for problem, script in (("A", "generate_comparison_figures.py"),
                                ("B", "generate_paper_comparison_figures.py")):
            spec = importlib.util.spec_from_file_location(
                f"plot_{problem}", ROOT / f"problem{problem}/scripts" / script
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.plotters[problem] = module

    def test_evaluated_cases_use_their_own_arrays(self):
        for problem, folder, prefix in (("A", "reference", "baseline"),
                                        ("B", "qs_control_moreiter", "qs-control")):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                plotter = self.plotters[problem]
                paper = plotter.load_case(None, folder, prefix)
                data = {key: value.copy() for key, value in paper[-1].items()}
                data["nwl"] *= 0.75
                if problem == "B":
                    data["iota"] += 0.01
                directory = Path(tmp)
                source = ROOT / f"problem{problem}/data" / folder
                file_prefix = prefix if problem == "A" else folder
                shutil.copyfile(source / f"{file_prefix}_eq.h5", directory / "equilibrium.h5")
                if problem == "B":
                    shutil.copyfile(source / f"{file_prefix}_offset_wall.h5", directory / "offset_wall.h5")
                np.savez(directory / "diagnostics.npz", **data)
                loaded = plotter.load_case(directory, "unused", "unused")
                for key, expected in data.items():
                    np.testing.assert_array_equal(loaded[-1][key], expected)
                self.assertEqual(loaded[0].NFP, paper[0].NFP)

    def test_incomplete_folder_does_not_fall_back_to_paper_data(self):
        for problem, plotter in self.plotters.items():
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(OSError):
                    plotter.load_case(Path(tmp), "reference", "baseline")

    def test_current_from_saved_equilibria_matches_paper_profiles(self):
        plotter = self.plotters["B"]
        rho = np.linspace(0, 1, 101)
        for folder, prefix, role in (("qs_control_moreiter", "qs-control", "reference"),
                                     ("stage_3", "stage-3", "nwl_aware")):
            eq, _, _ = plotter.load_case(None, folder, prefix)
            path = ROOT / f"problemB/results/{role}_output_current_profile_validation.json"
            metadata = json.loads(path.read_text())
            expected = metadata["scale"] * np.polynomial.polynomial.polyval(rho, metadata["underlying_params"])
            np.testing.assert_allclose(eq.current(rho), expected, rtol=1e-13, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
