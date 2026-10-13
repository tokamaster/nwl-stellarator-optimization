"""Regression tests for saving solver results without losing array values."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from solver_result import save_result


class SolverResultTests(unittest.TestCase):
    def test_numpy_and_native_values(self):
        result = dict(success=np.bool_(True), message="Converged", nit=np.int64(2),
                      nfev=3, fun=np.array([1.0, -2.0]), optimality=np.float64(0.25),
                      history=object())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            save_result(path, result)
            self.assertEqual(json.loads(path.read_text()), dict(
                success=True, message="Converged", nit=2, nfev=3,
                fun=[1.0, -2.0], optimality=0.25))

    @unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is not installed")
    def test_jax_solver_values(self):
        import jax.numpy as jnp

        result = dict(success=True, message="Iteration limit", nit=jnp.array(4),
                      fun=jnp.array([1.0, -2.0]), optimality=jnp.array(0.125))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            save_result(path, result)
            self.assertEqual(json.loads(path.read_text()), dict(
                success=True, message="Iteration limit", nit=4,
                fun=[1.0, -2.0], optimality=0.125))


if __name__ == "__main__":
    unittest.main()
