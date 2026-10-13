"""Small numerical regression tests for saved-data processing."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "scripts"))
from nwl_statistics import load_batches, statistics
from summarize import nwl_metrics, saved_rows
from plot_validation_comparison import comparison_rows
sys.path.insert(0, str(ROOT / "validation/scripts"))
from fusion_source import reaction_rate


class DataTests(unittest.TestCase):
    def test_missing_batch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in (0, 1, 3):
                p = Path(tmp) / f"batch{i}"
                p.mkdir()
                np.save(p / "nwl_mat.npy", np.full((2, 2), 10.0))
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_batches(tmp)

    def test_batch_uncertainty(self):
        batches = np.array([[[1.0]], [[3.0]]])
        result = statistics(batches)
        self.assertEqual(result["mean"].item(), 2.0)
        self.assertAlmostEqual(result["stderr"].item(), 1.0)
        self.assertAlmostEqual(result["relerr"].item(), 0.5)

    def test_saved_batch_aggregates(self):
        for case in ("axisymmetric", "iter", "twisty"):
            directory = ROOT / "validation" / case
            batches = load_batches(directory, 100)
            for name, actual in statistics(batches).items():
                np.testing.assert_allclose(actual, np.load(directory / f"nwl_{name}.npy"), rtol=1e-12, atol=1e-13)

    def test_area_fraction_is_weighted_and_strict(self):
        result = nwl_metrics([0.79, 0.80, 0.81], [1, 2, 7])
        self.assertAlmostEqual(result["area_fraction_above_0p80"], 0.7)
        self.assertAlmostEqual(result["average_MW_m2"], 0.806)
        with self.assertRaises(ValueError):
            nwl_metrics([0.8], [-1])

    def test_comparison_uses_selected_global_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            for case in ("axisymmetric", "iter", "twisty"):
                destination = data / case
                destination.mkdir()
                source = ROOT / "validation" / case
                for name in ("desc_nwl", "desc_theta", "desc_zeta", "nwl_mean", "poloidal_bins", "toroidal_bins", "area_mat"):
                    shutil.copyfile(source / f"{name}.npy", destination / f"{name}.npy")
                (destination / "desc_global_metrics.json").write_text(json.dumps({"neutron_power_MW": 20, "wall_area_m2": 5}))
            rows = comparison_rows(data)
            self.assertEqual([r["p_n_over_s_a"] for r in rows if r["method"] == "DESC"], [4, 4, 4])

    def test_fusion_source_units_and_zero(self):
        self.assertEqual(reaction_rate(1e20, 0), 0)
        self.assertEqual(reaction_rate(0, 10), 0)
        # At 10 keV Bosch-Hale DT reactivity is approximately 1.136e-16 cm^3/s.
        np.testing.assert_allclose(reaction_rate(1e20, 10), 2.8404e11, rtol=2e-4)

    def test_paper_summary(self):
        rows = saved_rows()
        np.testing.assert_allclose([r["peak_MW_m2"] for r in rows],
            [0.925927608218456, 0.7977810760395878, 0.9644113766874669, 0.7983251389771882], rtol=1e-12)
        np.testing.assert_allclose([r["area_fraction_above_0p80"] for r in rows],
            [0.07126829957444584, 0, 0.10798663420215336, 0], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
