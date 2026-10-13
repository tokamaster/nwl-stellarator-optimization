"""Write portable convergence information from a DESC solver result."""
import json

import numpy as np


def save_result(path, result):
    keys = ("success", "message", "nit", "nfev", "njev", "fun", "optimality")
    # np.asarray handles both NumPy and JAX scalars/arrays; tolist also preserves
    # ordinary strings and booleans as JSON-compatible Python values.
    payload = {key: np.asarray(result[key]).tolist() for key in keys if key in result}
    path.write_text(json.dumps(payload, indent=2) + "\n")
