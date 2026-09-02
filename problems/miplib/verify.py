"""Independent feasibility check of a MIPLIB solution.

Reads the instance with HiGHS but does all the arithmetic itself: variable bounds, integrality,
every row activity, and the objective (in the instance's own sense). Tolerance 1e-6 absolute-or-relative,
which is stricter than MIPLIB's own solution checker, so anything that passes here should pass there.

    python verify.py candidate.json      # {"target": name, "solution": {var: value}}
"""

import json
import os
import sys

import highspy
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import instance_path  # noqa: E402

TOL = 1e-6


def load(name):
    h = highspy.Highs()
    h.silent()
    if h.readModel(instance_path(name)) != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS could not read {name}")
    return h


def check(solution, name):
    h = load(name)
    lp = h.getLp()
    n, m = lp.num_col_, lp.num_row_
    names = list(lp.col_names_)
    idx = {nm: i for i, nm in enumerate(names)}
    unknown = [k for k in solution if k not in idx]
    x = np.zeros(n)
    for k, v in solution.items():
        if k in idx:
            x[idx[k]] = float(v)

    lo, up = np.array(lp.col_lower_), np.array(lp.col_upper_)
    tol = TOL * np.maximum(1.0, np.abs(x))
    bound_viol = max(0.0, float((lo - x - tol).max()), float((x - up - tol).max()))

    kinds = list(lp.integrality_)
    isint = np.array([k != highspy.HighsVarType.kContinuous for k in kinds], bool) if kinds else np.zeros(n, bool)
    int_viol = float(np.abs(x[isint] - np.round(x[isint])).max()) if isint.any() else 0.0

    A = lp.a_matrix_
    if A.format_ == highspy.MatrixFormat.kColwise:
        M = sp.csc_matrix((A.value_, A.index_, A.start_), shape=(m, n))
    else:
        M = sp.csr_matrix((A.value_, A.index_, A.start_), shape=(m, n))
    ax = M @ x
    rl, ru = np.array(lp.row_lower_), np.array(lp.row_upper_)
    tolr = TOL * np.maximum(1.0, np.abs(ax))
    row_viol = max(0.0, float((rl - ax - tolr).max()), float((ax - ru - tolr).max())) if m else 0.0

    obj = float(np.dot(np.array(lp.col_cost_), x) + lp.offset_)
    sense = "min" if h.getObjectiveSense()[1] == highspy.ObjSense.kMinimize else "max"
    feasible = not unknown and bound_viol <= 0 and int_viol <= TOL and row_viol <= 0
    return {
        "feasible": bool(feasible),
        "obj": obj,
        "sense": sense,
        "bound_viol": bound_viol,
        "int_viol": int_viol,
        "row_viol": row_viol,
        "unknown_vars": unknown[:5],
        "cols": n,
        "rows": m,
    }


def to_sol(solution, obj):
    """MIPLIB / SCIP .sol format: '=obj=' line, then 'name value' for every nonzero variable."""
    lines = [f"=obj= {obj:.15g}"]
    lines += [f"{k} {v:.15g}" for k, v in solution.items() if float(v) != 0.0]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    res = check(d["solution"], d["target"])
    print(json.dumps(res))
    sys.exit(0 if res["feasible"] else 1)
