"""Seed solver: multi-start penalty L-BFGS-B over (x, y, r), then LP-optimal radii for fixed centres,
then a tiny uniform shrink so the packing is strictly feasible with zero tolerance.
Interface: python solver.py --n N --time SECONDS --seed S --out PATH
Writes JSON {"n": N, "circles": [[x, y, r], ...]} in the unit square [0,1]^2.
"""

import argparse
import json
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog


def penalty(z, n, mu):
    c = z[: 2 * n].reshape(n, 2)
    r = z[2 * n :]
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diff**2).sum(-1) + 1e-18)
    v = np.maximum(0.0, r[:, None] + r[None, :] - d)
    np.fill_diagonal(v, 0.0)
    wv = np.maximum(
        0.0,
        np.stack([r - c[:, 0], r - (1 - c[:, 0]), r - c[:, 1], r - (1 - c[:, 1])], 1),
    )
    f = -r.sum() + mu * (0.5 * (v**2).sum() + (wv**2).sum())
    gr = -1.0 + mu * (2 * v.sum(1) + 2 * wv.sum(1))
    gc = mu * (-2 * ((v / d)[:, :, None] * diff).sum(1))
    gc[:, 0] += mu * (-2 * wv[:, 0] + 2 * wv[:, 1])
    gc[:, 1] += mu * (-2 * wv[:, 2] + 2 * wv[:, 3])
    return f, np.concatenate([gc.ravel(), gr])


def lp_radii(c):
    n = len(c)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    m = len(iu[0])
    A = np.zeros((m, n))
    A[np.arange(m), iu[0]] = 1
    A[np.arange(m), iu[1]] = 1
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1]))
    res = linprog(
        -np.ones(n),
        A_ub=A,
        b_ub=d[iu],
        bounds=[(0, max(w, 0)) for w in wall],
        method="highs",
    )
    return res.x if res.success else np.zeros(n)


def make_strict(c, r):
    """Shrink every radius by one delta so all constraints hold with margin in float arithmetic."""
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pv = (r[:, None] + r[None, :] - d).max()
    wv = (r - np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1]))).max()
    delta = max(pv / 2, wv, 0.0) + 1e-12
    return np.maximum(r - delta, 1e-9)


def feasible_sum(c, r):
    circ = [[float(x), float(y), float(rr)] for (x, y), rr in zip(c, r)]
    wall = min(min(x - rr, 1 - x - rr, y - rr, 1 - y - rr) for x, y, rr in circ)
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    s2 = (r[:, None] + r[None, :]) ** 2
    np.fill_diagonal(s2, -1.0)
    ok = wall >= 0 and (d2 - s2).min() >= 0 and (r > 0).all()
    return (float(r.sum()) if ok else -1.0), circ


def init(n, rng):
    if rng.random() < 0.5:
        c = rng.random((n, 2))
    else:  # jittered hex-ish grid
        k = math.ceil(math.sqrt(n))
        pts = np.array([[(j + 0.5 * (i % 2)) / k, i / k] for i in range(k + 1) for j in range(k + 1)])
        pts = pts[rng.permutation(len(pts))[:n]]
        c = np.clip(pts + rng.normal(0, 0.02, pts.shape), 0.02, 0.98)
    r = np.full(n, 0.4 / math.sqrt(n)) * rng.uniform(0.5, 1.5, n)
    return c, r


def solve(n, budget, seed):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = (-1.0, None)
    while time.time() - t0 < budget:
        c, r = init(n, rng)
        z = np.concatenate([c.ravel(), r])
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        for mu in (10, 100, 1e3, 1e4, 1e5):
            res = minimize(
                penalty,
                z,
                args=(n, mu),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300},
            )
            z = res.x
            if time.time() - t0 > budget:
                break
        c = z[: 2 * n].reshape(n, 2)
        r = make_strict(c, lp_radii(c))
        s, circ = feasible_sum(c, r)
        if s > best[0]:
            best = (s, circ)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s, circ = solve(a.n, a.time, a.seed)
    json.dump({"n": a.n, "circles": circ or [], "sum": s}, open(a.out, "w"))
