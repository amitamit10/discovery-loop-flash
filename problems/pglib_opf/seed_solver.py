"""Seed AC-OPF solver: PYPOWER's primal-dual interior point method (PIPS), then random multi-start.

Phase 1: solve the case from the file's start point with tight tolerances; re-solve with slightly shrunk limits if
the independent verifier rejects it at 1e-6.
Phase 2: until the time budget, restart PIPS from perturbed voltages/dispatch and keep the best verified solution.

    python seed_solver.py --target pglib_opf_case14_ieee --time 60 --seed 1 --out sol.json
writes {"target", "obj", "solution": {"vm", "va", "pg", "qg"}} (pu, radians), saved atomically on every improvement.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone use; the loop also sets PYTHONPATH
import matpower as mp  # noqa: E402
import verify  # noqa: E402
from records import case_path  # noqa: E402


def to_ppc(case):
    """MATPOWER dict -> PYPOWER case dict (gen padded to 21 and branch to 17 columns as PYPOWER expects)."""
    gen = np.zeros((case["gen"].shape[0], 21))
    gen[:, : case["gen"].shape[1]] = case["gen"][:, :21]
    branch = np.zeros((case["branch"].shape[0], 17))
    branch[:, : case["branch"].shape[1]] = case["branch"][:, :17]
    return {
        "version": "2",
        "baseMVA": case["baseMVA"],
        "bus": case["bus"].copy(),
        "gen": gen,
        "branch": branch,
        "gencost": case["gencost"].copy(),
    }


def shrink(ppc, eps):
    """Tighten every inequality by eps (pu / MW / MVA) so PIPS's own tolerance lands inside the verifier's."""
    p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()}
    base = p["baseMVA"]
    p["bus"][:, mp.VMAX] -= eps
    p["bus"][:, mp.VMIN] += eps
    on = p["gen"][:, mp.GEN_STATUS] > 0
    p["gen"][on, mp.PMAX] -= eps * base
    p["gen"][on, mp.PMIN] += eps * base
    p["gen"][on, mp.QMAX] -= eps * base
    p["gen"][on, mp.QMIN] += eps * base
    lim = p["branch"][:, mp.RATE_A] > 0
    p["branch"][lim, mp.RATE_A] -= eps * base
    return p


def solve(ppc, feastol=1e-7):
    """PIPS OPF, then a Newton power-flow polish so nodal balance holds to 1e-10 (PIPS's own tolerance is relative
    and leaves ~1e-5 pu residuals at buses with large reactive output). The polish keeps Pg and gen-bus Vm from the
    OPF, so cost is unchanged; the shrink() margin absorbs the tiny Qg / slack shifts it introduces."""
    from pypower.api import ppoption, runopf, runpf

    opt = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        OPF_ALG=560,
        PDIPM_FEASTOL=feastol,
        PDIPM_MAX_IT=500,
    )
    r = runopf(ppc, opt)
    if not r["success"]:
        return None
    pos = {int(b): i for i, b in enumerate(r["bus"][:, mp.BUS_I])}
    r["gen"][:, mp.VG] = [r["bus"][pos[int(b)], mp.VM] for b in r["gen"][:, mp.GEN_BUS]]
    pf, ok = runpf(r, ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10, PF_MAX_IT=50, ENFORCE_Q_LIMS=0))
    if ok:
        r = pf
    base = r["baseMVA"]
    return {
        "vm": r["bus"][:, mp.VM].tolist(),
        "va": np.deg2rad(r["bus"][:, mp.VA]).tolist(),
        "pg": (r["gen"][:, mp.PG] / base).tolist(),
        "qg": (r["gen"][:, mp.QG] / base).tolist(),
    }


def perturb(ppc, rng, scale):
    p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()}
    bus, gen = p["bus"], p["gen"]
    bus[:, mp.VM] = np.clip(bus[:, mp.VM] + rng.normal(0, 0.03 * scale, bus.shape[0]), bus[:, mp.VMIN], bus[:, mp.VMAX])
    bus[:, mp.VA] += rng.normal(0, 5.0 * scale, bus.shape[0])
    on = gen[:, mp.GEN_STATUS] > 0
    span = gen[on, mp.PMAX] - gen[on, mp.PMIN]
    gen[on, mp.PG] = np.clip(
        gen[on, mp.PG] + rng.normal(0, 0.3 * scale, on.sum()) * span, gen[on, mp.PMIN], gen[on, mp.PMAX]
    )
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--time", type=float, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    case = mp.load(case_path(a.target))
    ppc = to_ppc(case)
    best = None

    def save(sol, obj):
        tmp = a.out + ".tmp"
        json.dump({"target": a.target, "obj": obj, "solution": sol}, open(tmp, "w"))
        os.replace(tmp, a.out)

    def attempt(p):
        nonlocal best
        for eps in (0.0, 2e-6, 1e-5):
            sol = solve(shrink(p, eps) if eps else p)
            if sol is None:
                continue
            res = verify.check(sol, a.target)
            if res["feasible"]:
                if best is None or res["obj"] < best:
                    best = res["obj"]
                    save(sol, best)
                return res["obj"]
        return None

    attempt(ppc)
    tries = 0
    while time.time() - t0 < a.time - 5:
        tries += 1
        attempt(perturb(ppc, rng, scale=1.0 if tries % 3 else 2.5))
        if best is None and tries > 20:
            break
    print(f"best={best} tries={tries} secs={time.time() - t0:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
