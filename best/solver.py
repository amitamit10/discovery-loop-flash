"""Basin-hopping solver for csqv: penalty L-BFGS-B multi-start, then perturb-and-repolish of the incumbent
(jitter / relocate smallest circles into largest holes / swap), LP-optimal radii for fixed centres,
an exact SLSQP active-set polish on the contact graph, and a final strict shrink.
Interface: python solver.py --n N --time SECONDS --seed S --out PATH
"""

import argparse
import json
import math
import os
import time
import numpy as np
from scipy.optimize import minimize, linprog


class Timeout(Exception):
    pass


# ----------------------------------------------------------------------------- penalty phase
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


def run_penalty(c, r, n, mus, maxiter, deadline):
    z = np.concatenate([c.ravel(), r])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    for mu in mus:
        try:
            res = minimize(
                penalty,
                z,
                args=(n, mu),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter},
            )
            z = res.x
        except Exception:
            break
        if time.time() > deadline:
            break
    return z[: 2 * n].reshape(n, 2).copy(), z[2 * n :].copy()


# ----------------------------------------------------------------------------- LP radii / strictness
def lp_radii(c, r0=None):
    n = len(c)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    I0, J0 = np.triu_indices(n, 1)
    dd = d[I0, J0]
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1]))
    bounds = [(0.0, max(float(w), 0.0)) for w in wall]
    if r0 is None or len(I0) == 0:
        mask = np.ones(len(I0), bool)
    else:
        mask = dd - r0[I0] - r0[J0] < 0.5 * max(float(r0.mean()), 1e-6)
    opts = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
    for rnd in range(5):
        if rnd == 4:
            mask = np.ones(len(I0), bool)
        I = I0[mask]
        J = J0[mask]
        m = len(I)
        if m > 0:
            A = np.zeros((m, n))
            A[np.arange(m), I] = 1.0
            A[np.arange(m), J] = 1.0
            b = dd[mask]
        else:
            A, b = None, None
        res = None
        for o in (opts, None):
            try:
                res = linprog(
                    -np.ones(n),
                    A_ub=A,
                    b_ub=b,
                    bounds=bounds,
                    method="highs",
                    options=o,
                )
                if res.success:
                    break
            except Exception:
                res = None
        if res is None or not res.success:
            return np.zeros(n)
        r = np.maximum(res.x, 0.0)
        viol = (r[I0] + r[J0] - dd > 1e-9) & ~mask
        if not viol.any():
            return r
        mask |= viol
    return np.zeros(n)


def make_strict(c, r):
    r = np.asarray(r, float).copy()
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1]))
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    for _ in range(4):
        pv = (r[:, None] + r[None, :] - d).max()
        wv = (r - wall).max()
        delta = max(pv / 2, wv, 0.0) + 1e-12
        r = np.maximum(r - delta, 1e-9)
        if (r[:, None] + r[None, :] - d).max() < 0 and (r - wall).max() < 0:
            break
    return r


def feasible_sum(c, r):
    circ = [[float(x), float(y), float(rr)] for (x, y), rr in zip(c, r)]
    wall = min(min(x - rr, 1 - x - rr, y - rr, 1 - y - rr) for x, y, rr in circ)
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    s2 = (r[:, None] + r[None, :]) ** 2
    np.fill_diagonal(s2, -1.0)
    ok = wall >= 0 and (d2 - s2).min() >= 0 and (r > 0).all()
    return (float(r.sum()) if ok else -1.0), circ


def evaluate(c, r):
    r2 = make_strict(c, r)
    s, circ = feasible_sum(c, r2)
    return {"s": s, "circ": circ, "c": c.copy(), "r": r2}


def trivial(n):
    k = math.ceil(math.sqrt(n))
    r = 0.5 / k * (1 - 1e-6)
    c = np.array([[(i % k + 0.5) / k, (i // k + 0.5) / k] for i in range(n)])
    return evaluate(c, np.full(n, r))


# ----------------------------------------------------------------------------- SLSQP active-set polish
def polish(c, r, n, deadline):
    best = evaluate(c, r)
    if n < 2:
        return best
    z = np.concatenate([c.ravel(), r])
    I0, J0 = np.triu_indices(n, 1)
    rbar = max(float(r.mean()), 1e-6)
    scale = 1.0 / (2 * rbar)
    tol = 0.3 * rbar
    incl = np.zeros(len(I0), bool)
    wincl = np.zeros(4 * n, bool)
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n
    gobj = np.zeros(3 * n)
    gobj[2 * n :] = -1.0
    obj = lambda z: -z[2 * n :].sum()
    ojac = lambda z: gobj

    for _ in range(4):
        if time.time() > deadline:
            break
        C = z[: 2 * n].reshape(n, 2)
        R = z[2 * n :]
        gap = np.sqrt(((C[I0] - C[J0]) ** 2).sum(1)) - R[I0] - R[J0]
        incl |= gap < tol
        wg = np.concatenate([C[:, 0] - R, 1 - C[:, 0] - R, C[:, 1] - R, 1 - C[:, 1] - R])
        wincl |= wg < tol
        I = I0[incl]
        J = J0[incl]
        m = len(I)
        wI = np.nonzero(wincl)[0]
        side = wI // n
        wi = wI % n
        wcol = 2 * wi + (side >= 2)
        wsgn = np.where(side % 2 == 0, 1.0, -1.0)
        wconst = np.where(side % 2 == 1, 1.0, 0.0)
        mw = len(wI)
        rows = np.arange(m)

        def cons(z):
            C = z[: 2 * n].reshape(n, 2)
            R = z[2 * n :]
            dx = C[I, 0] - C[J, 0]
            dy = C[I, 1] - C[J, 1]
            g1 = (dx * dx + dy * dy - (R[I] + R[J]) ** 2) * scale
            g2 = wconst + wsgn * z[wcol] - R[wi]
            return np.concatenate([g1, g2])

        def cjac(z):
            C = z[: 2 * n].reshape(n, 2)
            R = z[2 * n :]
            dx = C[I, 0] - C[J, 0]
            dy = C[I, 1] - C[J, 1]
            Jm = np.zeros((m + mw, 3 * n))
            Jm[rows, 2 * I] = 2 * dx * scale
            Jm[rows, 2 * J] = -2 * dx * scale
            Jm[rows, 2 * I + 1] = 2 * dy * scale
            Jm[rows, 2 * J + 1] = -2 * dy * scale
            s = -2 * (R[I] + R[J]) * scale
            Jm[rows, 2 * n + I] = s
            Jm[rows, 2 * n + J] = s
            wr = m + np.arange(mw)
            Jm[wr, wcol] = wsgn
            Jm[wr, 2 * n + wi] = -1.0
            return Jm

        last = [z.copy()]

        def cb(xk):
            last[0] = xk.copy()
            if time.time() > deadline:
                raise Timeout

        try:
            res = minimize(
                obj,
                z,
                jac=ojac,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                callback=cb,
                options={"maxiter": 120, "ftol": 1e-14},
            )
            znew = res.x
        except Timeout:
            znew = last[0]
        except Exception:
            break
        C2 = znew[: 2 * n].reshape(n, 2).copy()
        R2 = znew[2 * n :].copy()
        for rr in (R2, lp_radii(C2, R2)):
            cand = evaluate(C2, rr)
            if cand["s"] > best["s"]:
                best = cand
        gap2 = np.sqrt(((C2[I0] - C2[J0]) ** 2).sum(1)) - R2[I0] - R2[J0]
        bad = (gap2 < -1e-9) & ~incl
        wg2 = np.concatenate([C2[:, 0] - R2, 1 - C2[:, 0] - R2, C2[:, 1] - R2, 1 - C2[:, 1] - R2])
        wbad = (wg2 < -1e-9) & ~wincl
        z = znew
        if not bad.any() and not wbad.any():
            break
        incl |= bad
        wincl |= wbad
    return best


# ----------------------------------------------------------------------------- init & moves
def init(n, rng):
    u = rng.random()
    if u < 0.4:
        c = rng.random((n, 2))
    elif u < 0.75:
        k = math.ceil(math.sqrt(n))
        pts = np.array([[(j + 0.5 * (i % 2)) / k, i / k] for i in range(k + 1) for j in range(k + 1)])
        pts = pts[rng.permutation(len(pts))[:n]]
        c = np.clip(pts + rng.normal(0, 0.02, pts.shape), 0.02, 0.98)
    else:
        k = math.ceil(math.sqrt(n))
        pts = np.array([[(j + 0.5) / k, (i + 0.5) / k] for i in range(k) for j in range(k)])
        pts = pts[rng.permutation(len(pts))[:n]]
        c = np.clip(pts + rng.normal(0, 0.02, pts.shape), 0.02, 0.98)
    r = np.full(n, 0.4 / math.sqrt(n)) * rng.uniform(0.5, 1.5, n)
    return c, r


def relocate(c, r, rng, k):
    n = len(c)
    rbar = max(float(r.mean()), 1e-6)
    idx = np.argsort(r + rng.normal(0, 0.3 * rbar, n))[:k]
    keep = np.ones(n, bool)
    keep[idx] = False
    ck, rk = c[keep], r[keep]
    P = rng.random((3000, 2))
    wall = np.minimum(np.minimum(P[:, 0], 1 - P[:, 0]), np.minimum(P[:, 1], 1 - P[:, 1]))
    if len(ck):
        h = np.minimum(
            (np.sqrt(((P[:, None, :] - ck[None, :, :]) ** 2).sum(-1)) - rk[None, :]).min(1),
            wall,
        )
    else:
        h = wall
    c2, r2 = c.copy(), r.copy()
    for i in idx:
        j = int(np.argmax(h))
        hr = max(float(h[j]), 1e-3)
        c2[i] = P[j]
        r2[i] = hr
        h = np.minimum(h, np.sqrt(((P - P[j]) ** 2).sum(1)) - hr)
    return c2, r2


def perturb(c, r, rng, n):
    rbar = max(float(r.mean()), 1e-6)
    u = rng.random()
    c2, r2 = c.copy(), r.copy()
    if u < 0.35:
        sig = rbar * rng.choice([0.05, 0.15, 0.4])
        c2 += rng.normal(0, sig, c2.shape)
        mus = (1e3, 1e4, 1e5)
    elif u < 0.55:
        k = max(1, int(n * rng.uniform(0.1, 0.3)))
        idx = rng.choice(n, k, replace=False)
        c2[idx] += rng.normal(0, rbar * 0.8, (k, 2))
        mus = (1e2, 1e3, 1e4, 1e5)
    elif u < 0.85:
        k = int(rng.integers(1, min(4, n) + 1))
        c2, r2 = relocate(c, r, rng, k)
        mus = (1e2, 1e3, 1e4, 1e5)
    else:
        order = np.argsort(r)
        q = max(1, n // 3)
        i = order[rng.integers(0, q)]
        j = order[n - 1 - rng.integers(0, q)]
        c2[[i, j]] = c2[[j, i]]
        mus = (1e2, 1e3, 1e4, 1e5)
    c2 = np.clip(c2, 1e-3, 1 - 1e-3)
    return c2, r2, mus


# ----------------------------------------------------------------------------- driver
def save(out, n, best):
    data = {"n": n, "circles": best["circ"], "sum": best["s"]}
    try:
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, out)
    except Exception:
        with open(out, "w") as f:
            json.dump(data, f)


def solve(n, budget, seed, out):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    reserve = min(5.0, max(0.5, 0.05 * budget))
    end = t0 + budget - reserve
    p1_end = t0 + 0.3 * budget
    p2_end = t0 + 0.8 * budget - reserve
    best = trivial(n)
    save(out, n, best)
    if n == 1:
        best = evaluate(np.array([[0.5, 0.5]]), np.array([0.5]))
        save(out, n, best)
        return best

    # Phase 1: cold multi-start
    starts = 0
    while time.time() < p1_end or starts < 2:
        if time.time() > end:
            break
        c, r = init(n, rng)
        c, r = run_penalty(c, r, n, (10, 100, 1e3, 1e4, 1e5), 300, end)
        cand = evaluate(c, lp_radii(c, r))
        starts += 1
        if cand["s"] > best["s"]:
            best = cand
            save(out, n, best)

    if time.time() < end and best["s"] > 0:
        cand = polish(best["c"], best["r"], n, min(end, time.time() + 0.15 * budget))
        if cand["s"] > best["s"]:
            best = cand
            save(out, n, best)

    # Phase 2: basin hopping on the incumbent
    cur = best
    stall = 0
    last_polish = time.time()
    while time.time() < p2_end and cur["s"] > 0:
        c, r, mus = perturb(cur["c"], cur["r"], rng, n)
        c, r = run_penalty(c, r, n, mus, 200, end)
        cand = evaluate(c, lp_radii(c, r))
        prog = (time.time() - t0) / max(budget, 1e-9)
        T = 3e-4 * max(0.0, 1 - prog)
        if cand["s"] > 0 and cand["s"] > cur["s"] - T * rng.exponential():
            cur = cand
        if cand["s"] > best["s"]:
            best = cand
            cur = best
            stall = 0
            save(out, n, best)
            if (n <= 60 or time.time() - last_polish > 0.15 * budget) and time.time() < p2_end:
                pol = polish(best["c"], best["r"], n, min(p2_end, time.time() + 0.1 * budget))
                last_polish = time.time()
                if pol["s"] > best["s"]:
                    best = pol
                    cur = best
                    save(out, n, best)
        else:
            stall += 1
            if stall > 40:
                cur = best
                stall = 0

    # Phase 3: final polish of the best basin
    if time.time() < end and best["s"] > 0:
        pol = polish(best["c"], best["r"], n, end)
        if pol["s"] > best["s"]:
            best = pol
            save(out, n, best)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    best = solve(a.n, a.time, a.seed, a.out)
    save(a.out, a.n, best)
