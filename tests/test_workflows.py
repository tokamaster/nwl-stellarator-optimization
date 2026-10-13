"""Check continuation order and failure handling without running an optimizer."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PARENTS = {
    "A": {"reference": None, "peaking-0p5": "reference", "peaking-1p0": "peaking-0p5",
          "peaking-1p25": "peaking-1p0", "qs-100": "peaking-1p25",
          "qs-110": "qs-100", "nwl-aware": "qs-110"},
    "B": {"initialize": None, "qs-control": "initialize", "reference": "qs-control",
          "cap-0p880": "initialize", "cap-0p865": "cap-0p880", "parent": "cap-0p865",
          "stage-1": "parent", "stage-2": "stage-1", "nwl-aware": "stage-2"},
}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.calls = self.directory / "calls.jsonl"
        stub = self.directory / "record_calls.py"
        stub.write_text('''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[1] == os.environ.get("FAIL_STAGE"):
    sys.exit(9)
if "--input-eq" in args:
    assert Path(args[args.index("--input-eq") + 1]).is_file()
if Path(args[0]).name == "evaluate.py":
    assert Path(args[2]).is_file()
if "--reference-dir" in args:
    for flag in ("--reference-dir", "--nwl-aware-dir"):
        assert (Path(args[args.index(flag) + 1]) / "diagnostics.npz").is_file()
out = Path(args[args.index("--output-dir") + 1])
out.mkdir()
for name in ("equilibrium.h5", "offset_wall.h5", "diagnostics.npz"):
    (out / name).touch()
''')
        launcher = self.directory / "mock python"
        launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(stub))} "$@"\n')
        launcher.chmod(0o755)
        self.env = dict(os.environ, PYTHON=str(launcher), CALL_LOG=str(self.calls))

    def run_workflow(self, problem, fail_stage=""):
        return subprocess.run(
            ["sh", str(ROOT / f"problem{problem}/scripts/reproduce.sh"),
             str(self.directory / f"new results {problem}")], cwd=self.directory,
            env=dict(self.env, FAIL_STAGE=fail_stage), capture_output=True, text=True,
        )

    def test_complete_chains_use_only_new_outputs_and_refuse_overwrite(self):
        for problem, parents in PARENTS.items():
            with self.subTest(problem=problem):
                self.calls.write_text("")
                result = self.run_workflow(problem)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
                stages = [args for args in calls if Path(args[0]).name == "optimize.py"]
                self.assertEqual([args[1] for args in stages], list(parents))
                output = self.directory / f"new results {problem}"
                for args in stages:
                    parent = parents[args[1]]
                    if parent is None:
                        self.assertNotIn("--input-eq", args)
                    else:
                        self.assertEqual(Path(args[args.index("--input-eq") + 1]),
                                         output / parent / "equilibrium.h5")
                diagnostics = [args for args in calls if Path(args[0]).name == "evaluate.py"]
                self.assertEqual([Path(args[2]) for args in diagnostics],
                                 [output / name / "equilibrium.h5" for name in ("reference", "nwl-aware")])
                self.assertTrue((output / "figures").is_dir())
                before = self.calls.read_text()
                self.assertNotEqual(self.run_workflow(problem).returncode, 0)
                self.assertEqual(self.calls.read_text(), before)

    def test_failed_stage_stops_its_branch_and_diagnostics(self):
        for problem, stage in (("A", "peaking-0p5"), ("B", "cap-0p880")):
            with self.subTest(problem=problem):
                self.calls.write_text("")
                self.assertEqual(self.run_workflow(problem, stage).returncode, 9)
                calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
                self.assertEqual(calls[-1][1], stage)
                self.assertTrue(all(Path(args[0]).name == "optimize.py" for args in calls))


if __name__ == "__main__":
    unittest.main()
