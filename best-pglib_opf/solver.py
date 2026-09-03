"""AC-OPF solver: PIPS interior point + basin-hopping multi-start + targeted feasibility polish.

Phase 1: PIPS from the file's start point (and a flat start); Newton power-flow polish; if the independent
verifier rejects the point at 1e-6, re-solve warm-started from it with ONLY the near-binding constraints tightened
(uniform shrinking of every limit costs objective; targeted shrinking costs almost nothing).
Phase 2: until the time budget, restart PIPS from (a) Latin-hypercube samples over generator voltage set-points and
dispatch, (b) small/large perturbations of the incumbent (basin hopping with adaptive step), keeping the reference
angle at zero so every restart is verifiable. Best verified solution is saved atomically on every improvement.

    python solver.py --target pglib_opf_case14_ieee --time 60 --seed 1 --out sol.json
"""

import argparse
import json
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone use; the loop also sets PYTHONPATH
import matpower as mp  # noqa: E402
import verify  # noqa: E402
from records import case_path  # noqa: E402

from pypower.api import ppoption, runopf, runpf  # noqa: E402
from pypower.idx_bus import BUS_I, BUS_TYPE, VM, VA, VMAX, VMIN  # noqa: E402
from pypower.idx_gen import GEN_BUS, PG, QG, VG, PMAX, PMIN, QMAX, QMIN, GEN_STATUS  # noqa: E402
from pypower.idx_brch import F_BUS, T_BUS, RATE_A, ANGMIN, ANGMAX, PF, QF, PT, QT  # noqa: E402

np.seterr(all="ignore")


# ----------------------------------------------------------------------------------------------------- case handling
def to_ppc(case):
    """MATPOWER dict -> PYPOWER case dict (gen padded to 21 and branch to 17 columns as PYPOWER expects)."""
    gen = np.zeros((case["gen"].shape[0], 21))
    gen[:, : case["gen"].shape[1]] = case["gen"][:, :21]
    off = gen[:, GEN_STATUS] <= 0
    gen[off, PG] = 0.0
    gen[off, QG] = 0.0
    branch = np.zeros((case["branch"].shape[0], 17))
    branch[:, : case["branch"].shape[1]] = case["branch"][:, :17]
    return {
        "version": "2",
        "baseMVA": float(case["baseMVA"]),
        "bus": case["bus"].copy(),
        "gen": gen,
        "branch": branch,
        "gencost": case["gencost"].copy(),
    }


def cp(ppc):
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()}


def warm(ppc, r):
    """Copy of ppc (original limits) whose start point is taken from result r."""
    p = cp(ppc)
    p["bus"][:, VM] = r["bus"][:, VM]
    p["bus"][:, VA] = r["bus"][:, VA]
    p["gen"][:, PG] = r["gen"][:, PG]
    p["gen"][:, QG] = r["gen"][:, QG]
    p["gen"][:, VG] = r["gen"][:, VG]
    return p


def tighten(ppc, r, eps, tol=3e-5):
    """Shrink by eps (pu) only the inequality limits that the solution r sits within tol of (or violates)."""
    p = cp(ppc)
    base = p["baseMVA"]
    bus, gen, br = p["bus"], p["gen"], p["branch"]
    vm = r["bus"][:, VM]
    span_ok = (bus[:, VMAX] - bus[:, VMIN]) > 4 * eps
    sel = span_ok & (vm > bus[:, VMAX] - tol)
    bus[sel, VMAX] -= eps
    sel = span_ok & (vm < bus[:, VMIN] + tol)
    bus[sel, VMIN] += eps
    on = gen[:, GEN_STATUS] > 0
    e = eps * base
    t = tol * base
    for lo, hi, col in ((PMIN, PMAX, PG), (QMIN, QMAX, QG)):
        val = r["gen"][:, col]
        ok = on & ((gen[:, hi] - gen[:, lo]) > 4 * e)
        sel = ok & (val > gen[:, hi] - t)
        gen[sel, hi] -= e
        sel = ok & (val < gen[:, lo] + t)
        gen[sel, lo] += e
    if r["branch"].shape[1] > QT:
        s = np.maximum(
            np.hypot(r["branch"][:, PF], r["branch"][:, QF]), np.hypot(r["branch"][:, PT], r["branch"][:, QT])
        )
        lim = br[:, RATE_A] > 0
        sel = lim & (s > br[:, RATE_A] - t)
        br[sel, RATE_A] -= e
    # angle-difference limits (degrees); verifier tolerance 1e-6 deg
    pos = {int(b): i for i, b in enumerate(r["bus"][:, BUS_I])}
    fi = np.array([pos[int(b)] for b in br[:, F_BUS]])
    ti = np.array([pos[int(b)] for b in br[:, T_BUS]])
    ang = r["bus"][fi, VA] - r["bus"][ti, VA]
    ed = eps * 100.0
    td = tol * 100.0
    amax, amin = br[:, ANGMAX], br[:, ANGMIN]
    act = (amax - amin) > 4 * ed
    sel = act & (amax < 360) & (amax != 0) & (ang > amax - td)
    br[sel, ANGMAX] -= ed
    sel = act & (amin > -360) & (amin != 0) & (ang < amin + td)
    br[sel, ANGMIN] += ed
    return p


# ----------------------------------------------------------------------------------------------------------- solving
def pips(ppc, max_it, feastol=1e-7):
    """PIPS OPF then a Newton power-flow polish (nodal balance to 1e-10, Pg and gen-bus Vm kept => cost unchanged)."""
    opt = ppoption(VERBOSE=0, OUT_ALL=0, OPF_ALG=560, PDIPM_FEASTOL=feastol, PDIPM_MAX_IT=max_it)
    try:
        with redirect_stdout(sys.stderr):
            r = runopf(ppc, opt)
    except Exception:
        return None
    if not r or not r.get("success"):
        return None
    try:
        pos = {int(b): i for i, b in enumerate(r["bus"][:, BUS_I])}
        r["gen"][:, VG] = [r["bus"][pos[int(b)], VM] for b in r["gen"][:, GEN_BUS]]
        with redirect_stdout(sys.stderr):
            pf, ok = runpf(r, ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10, PF_MAX_IT=50, ENFORCE_Q_LIMS=0))
        if ok:
            r = pf
    except Exception:
        pass
    return r


def extract(r):
    base = r["baseMVA"]
    va = np.deg2rad(r["bus"][:, VA])
    ref = np.where(r["bus"][:, BUS_TYPE] == 3)[0]
    if len(ref):
        va = va - va[ref[0]]
    pg = r["gen"][:, PG] / base
    qg = r["gen"][:, QG] / base
    off = r["gen"][:, GEN_STATUS] <= 0
    pg[off] = 0.0
    qg[off] = 0.0
    return {
        "vm": r["bus"][:, VM].tolist(),
        "va": va.tolist(),
        "pg": pg.tolist(),
        "qg": qg.tolist(),
    }


# ------------------------------------------------------------------------------------------------------ start points
def fix_ref(p):
    ref = np.where(p["bus"][:, BUS_TYPE] == 3)[0]
    if len(ref):
        p["bus"][:, VA] -= p["bus"][ref[0], VA]
    return p


def perturb(ppc, r, rng, scale):
    """Perturb the state of result r (or ppc's own start point) by a relative scale; angles keep ref = 0."""
    p = warm(ppc, r) if r is not None else cp(ppc)
    bus, gen = p["bus"], p["gen"]
    nb = bus.shape[0]
    bus[:, VM] = np.clip(bus[:, VM] + rng.normal(0, 0.03 * scale, nb), bus[:, VMIN], bus[:, VMAX])
    bus[:, VA] += rng.normal(0, 5.0 * scale, nb)
    on = gen[:, GEN_STATUS] > 0
    span = gen[on, PMAX] - gen[on, PMIN]
    gen[on, PG] = np.clip(gen[on, PG] + rng.normal(0, 0.3 * scale, on.sum()) * span, gen[on, PMIN], gen[on, PMAX])
    pos = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    gen[:, VG] = [bus[pos[int(b)], VM] for b in gen[:, GEN_BUS]]
    return fix_ref(p)


def lhs_start(ppc, u):
    """Start from a Latin-hypercube row u in [0,1]^(2*ngen_on): generator voltage set-points and dispatch."""
    p = cp(ppc)
    bus, gen = p["bus"], p["gen"]
    on = np.where(gen[:, GEN_STATUS] > 0)[0]
    n = len(on)
    uv, up = u[:n], u[n : 2 * n]
    pos = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    gb = np.array([pos[int(b)] for b in gen[on, GEN_BUS]])
    lo = bus[gb, VMIN] + 0.01
    hi = bus[gb, VMAX] - 0.01
    vset = np.clip(lo + uv * (hi - lo), bus[gb, VMIN], bus[gb, VMAX])
    bus[:, VM] = np.clip(1.0 + 0.0 * bus[:, VM], bus[:, VMIN], bus[:, VMAX])
    bus[gb, VM] = vset
    bus[:, VA] = 0.0
    gen[on, VG] = vset
    gen[on, PG] = gen[on, PMIN] + up * (gen[on, PMAX] - gen[on, PMIN])
    return fix_ref(p)


def flat_start(ppc):
    p = cp(ppc)
    bus, gen = p["bus"], p["gen"]
    bus[:, VM] = np.clip(1.0, bus[:, VMIN], bus[:, VMAX])
    bus[:, VA] = 0.0
    on = gen[:, GEN_STATUS] > 0
    gen[on, PG] = 0.5 * (gen[on, PMIN] + gen[on, PMAX])
    gen[:, VG] = 1.0
    return p


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--time", type=float, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + max(a.time - 6.0, 3.0)
    rng = np.random.default_rng(a.seed)
    case = mp.load(case_path(a.target))
    ppc = to_ppc(case)
    best = None
    best_r = None

    def save(sol, obj):
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"target": a.target, "obj": obj, "solution": sol}, f)
        os.replace(tmp, a.out)

    def check(r):
        nonlocal best, best_r
        try:
            sol = extract(r)
            res = verify.check(sol, a.target)
        except Exception:
            return None
        if not res.get("feasible"):
            return None
        obj = float(res["obj"])
        if best is None or obj < best:
            best, best_r = obj, r
            save(sol, obj)
        return obj

    def attempt(p, max_it):
        """PIPS from start p; on verifier rejection, warm re-solve with targeted tightening of near-binding limits."""
        r = pips(p, max_it)
        if r is None:
            return None
        obj = check(r)
        if obj is not None:
            return obj
        for eps in (6e-7, 3e-6, 1.5e-5, 6e-5):
            if time.time() > deadline:
                return None
            q = tighten(warm(p, r), r, eps)
            r2 = pips(q, max_it)
            if r2 is None:
                continue
            obj = check(r2)
            if obj is not None:
                return obj
            r = r2
        return None

    # Phase 1: file start, then flat start (often a different basin)
    max_att = 0.0
    ts = time.time()
    attempt(ppc, 500)
    max_att = max(max_att, time.time() - ts)
    if time.time() + 1.3 * max_att < deadline:
        ts = time.time()
        attempt(flat_start(ppc), 300)
        max_att = max(max_att, time.time() - ts)

    # Phase 2: basin hopping + LHS restarts
    non = int((ppc["gen"][:, GEN_STATUS] > 0).sum())
    try:
        from scipy.stats import qmc

        lhs = qmc.LatinHypercube(d=2 * non, seed=int(a.seed)).random(128)
    except Exception:
        lhs = rng.random((128, 2 * non))
    k_lhs = 0
    tries = 0
    stagnant = 0
    hop_scale = 0.25
    while time.time() + 1.3 * max_att + 1.0 < deadline:
        tries += 1
        before = best
        mode = tries % 4
        if best_r is None:
            p = perturb(ppc, None, rng, 1.0 if mode else 2.5)
        elif mode in (1, 2):
            p = perturb(ppc, best_r, rng, hop_scale)
        elif mode == 3:
            p = lhs_start(ppc, lhs[k_lhs % len(lhs)])
            k_lhs += 1
        else:
            p = perturb(ppc, best_r, rng, 2.0)
        ts = time.time()
        attempt(p, 250)
        max_att = max(max_att, time.time() - ts)
        if best is not None and (before is None or best < before - 1e-9):
            stagnant = 0
            hop_scale = max(0.1, hop_scale * 0.8)
        else:
            stagnant += 1
            if stagnant % 3 == 0:
                hop_scale = min(1.5, hop_scale * 1.5)
        if best is None and tries > 25:
            break
    print(f"best={best} tries={tries} secs={time.time() - t0:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
