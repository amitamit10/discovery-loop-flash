"""Independent check of a heuristic's output: feasibility at 1e-6 (shared with problems/miplib) plus the relative
primal gap to the proven optimum, in minimisation sense so that lower is always better.

    python verify.py candidate.json      # {"target": name, "solution": {var: value}}
"""

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import records  # noqa: E402


def _sibling(name):
    spec = importlib.util.spec_from_file_location("miplib_" + name, os.path.join(records.MIPLIB, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_V = _sibling("verify")
to_sol = _V.to_sol


def gap(obj, optimum, sense):
    """Relative primal gap >= 0 for any feasible point (negative would mean the .solu optimum is wrong)."""
    d = (obj - optimum) if sense == "min" else (optimum - obj)
    return d / max(1.0, abs(optimum))


def check(solution, name):
    res = _V.check(solution, name)
    res["optimum"] = records.opt(name)
    res["gap"] = gap(res["obj"], res["optimum"], res["sense"]) if res["feasible"] else None
    return res


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    res = check(d["solution"], d["target"])
    print(json.dumps(res))
    sys.exit(0 if res["feasible"] else 1)
