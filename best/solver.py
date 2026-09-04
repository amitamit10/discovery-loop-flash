IDEA: Corner-anchored variable-radii templates with square-lattice bank, binary-search minimal shrink and radii-rescale anneal moves
```python
"""Island-model basin-hopping solver for csqv: hex-lattice template inits (incl. under-full lattice + strip fill),
affine-lattice template bank (rotated / sheared / anisotropic hex + square lattices sized by bisection to hold exactly n sites),
corner-anchored templates with wall-biased selection, sparse-neighbour penalty L-BFGS-B, adaptive move selection incl.
lattice-slip row/band moves, local grain rotation, radius-rescale anneal and boundary-push moves,
half-plane crossover with a dihedral-symmetry-augmented elite pool, defect-migration moves with exact hole finding
(Delaunay circumcentres of wall-mirrored centres + samples, Nelder-Mead refined) and multi-step vacancy-diffusion
chains (remove weakest/loosest/neighbour-of-last-hole -> relax -> refill, repeated; insert -> relax -> evict weakest),
LP-optimal radii for fixed centres, fast exact KKT-Newton active-set polish on the contact graph for every candidate,
SLSQP polish on new bests, binary-search minimal-loss final strict shrink, and a second hopping phase that reclaims the tail of the
time budget after the final polish.
Independent chains run in worker processes (one per available core) and exchange elites through atomic JSON files.
Interface: python solver.py --n N --time SECONDS --seed S --out PATH
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

try:
    from scipy.spatial import Delaunay
except Exception:  # pragma: no cover
    Delaunay = None


class Timeout(Exception):
    pass


# ----------------------------------------------------------------------------- penalty phase
def penalty(z, n, mu, I, J):
    c = z[: 2 * n].reshape(n, 2)
    r = z[2 * n :]
    dx = c[I, 0] - c[J, 0]
    dy = c[I, 1] - c[J, 1]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    v = np.maximum(0.0, r[I] + r[J] - d)
    wv = np.maximum(0.0, np.stack([r - c[:, 0], r - (1 - c[:, 0]), r - c[:, 1], r - (1 - c[:, 1])], 1))
    f = -r.sum() + mu * ((v * v).sum() + (wv * wv).sum())
    vs = np.bincount(I, v, n) + np.bincount(J, v, n)
    gr = -1.0 + 2.0 * mu * (vs + wv.sum(1))
    k = 2.0 * mu * v / d
    kx = k * dx
    ky = k * dy
    gx = np.bincount(J, kx, n) - np.bincount(I, kx, n)
    gy = np.bincount(J, ky, n) - np.bincount(I, ky, n)
    gx += mu * (-2 * wv[:, 0] + 2 * wv[:, 1])
    gy += mu * (-2 * wv[:, 2] + 2 * wv[:, 3])
    g = np.empty(3 * n)
    g[0 : 2 * n : 2] = gx
    g[1 : 2 * n : 2] = gy
    g[2 * n :] = gr
    return f, g


def build_pairs(c, r, cut):
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    M = np.triu(d < (r[:, None] + r[None, :]) + cut, 1)
    I, J = np.nonzero(M)
    return I, J


def run_penalty(c, r, n, mus, maxiter, deadline):
    z = np.concatenate([c.ravel(), r])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    dense = n <= 40
    I0, J0 = np.triu_indices(n, 1)
    chunk = maxiter if dense else 50
    for mu in mus:
        it = 0
        while it < maxiter:
            if time.time() > deadline:
                return z[: 2 * n].reshape(n, 2).copy(), z[2 * n :].copy()
            if dense:
                I, J = I0, J0
            else:
                C = z[: 2 * n].reshape(n, 2)
                R = z[2 * n :]
                rbar = max(float(R.mean()), 1e-6)
                I, J = build_pairs(C, R, 3.0 * rbar + 0.02)
                if len(I) == 0:
                    I, J = I0[:1], J0[:1]
            m = min(chunk, maxiter - it)
            try:
                res = minimize(
                    penalty, z, args=(n, mu, I, J), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": m}
                )
                z = res.x
                it += m
                if res.nit < m:
                    break
            except Exception:
                return z[: 2 * n].reshape(n, 2).copy(), z[2 * n :].copy()
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
                res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method="highs", options=o)
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


def strict_ok(c, r):
    """Strict float64 feasibility, checked both in squared and in plain-distance form (conservative)."""
    try:
        if not np.isfinite(r).all() or not (r > 0).all():
            return False
        wall = np.minimum(np.minimum(c[:, 0] - r, 1 - c[:, 0] - r), np.minimum(c[:, 1] - r, 1 - c[:, 1] - r))
        if wall.min() < 0:
            return False
        d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        s = r[:, None] + r[None, :]
        s2 = s * s
        np.fill_diagonal(s2, -1.0)
        np.fill_diagonal(s, -1.0)
        if (d2 - s2).min() < 0:
            return False
        if (np.sqrt(d2) - s).min() < 0:
            return False
        return True
    except Exception:
        return False


def make_strict(c, r):
    r = np.asarray(r, float).copy()
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1]))
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pv = (r[:, None] + r[None, :] - d).max()
    wv = (r - wall).max()
    base = max(pv / 2, wv, 0.0)
    # binary-search minimal uniform shrink that is strictly feasible (recovers ~1e-12 per N)
    lo = 0.0
    hi = None
    # find hi that is feasible
    for eps in (1e-15, 5e-15, 1e-14, 5e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        r2 = np.maximum(r - (base + eps), 1e-9)
        if strict_ok(c, r2):
            hi = eps
            break
    if hi is None:
        hi = 1e-8
    # binary refine between lo and hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        r2 = np.maximum(r - (base + mid), 1e-9)
        if strict_ok(c, r2):
            hi = mid
        else:
            lo = mid
    r2 = np.maximum(r - (base + hi), 1e-9)
    if strict_ok(c, r2):
        return r2
    # fallback iterative
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


def load_candidate(path, n):
    """Read a packing JSON written by this solver (or a sibling worker); return a verified solution dict or None."""
    try:
        with open(path) as f:
            data = json.load(f)
        circ = data.get("circles")
        if not circ or len(circ) != n:
            return None
        arr = np.array(circ, float)
        if arr.shape != (n, 3) or not np.isfinite(arr).all():
            return None
        c = arr[:, :2].copy()
        r = arr[:, 2].copy()
        s, circ2 = feasible_sum(c, r)
        if s <= 0:
            return None
        return {"s": s, "circ": circ2, "c": c, "r": r}
    except Exception:
        return None


# ----------------------------------------------------------------------------- KKT-Newton active-set polish
def newton_polish(c, r, n, deadline, rounds=4, iters=10):
    """Exact polish of a near-optimal packing: treat the near-active contact/wall constraints as equalities and run
    Newton on the KKT system (linear objective, quadratic constraints -> constant Lagrangian Hessian). Constraints
    with negative multipliers are dropped and newly violated ones added between rounds. Always finished by LP radii
    and a strict feasibility check, so a bad step can only waste time, never produce an infeasible result."""
    best = evaluate(c, r)
    if n < 2 or time.time() > deadline:
        return best
    try:
        n2, n3 = 2 * n, 3 * n
        z = np.concatenate([c.ravel(), r]).astype(float)
        I0, J0 = np.triu_indices(n, 1)
        rbar = max(float(r.mean()), 1e-6)
        tol = 2.5e-3 * rbar
        gf = np.zeros(n3)
        gf[n2:] = -1.0
        C = z[:n2].reshape(n, 2)
        R = z[n2:]
        gap = np.sqrt(((C[I0] - C[J0]) ** 2).sum(1)) - R[I0] - R[J0]
        act = gap < tol
        wg = np.concatenate([C[:, 0] - R, 1 - C[:, 0] - R, C[:, 1] - R, 1 - C[:, 1] - R])
        wact = wg < tol
        rho = 1e-8
        delta = 1e-10
        idx3 = np.arange(n3)
        for _rnd in range(rounds):
            if time.time() > deadline:
                break
            I = I0[act]
            J = J0[act]
            m = len(I)
            wI = np.nonzero(wact)[0]
            side = wI // n
            wi = wI % n
            wcol = 2 * wi + (side >= 2)
            wsgn = np.where(side % 2 == 0, 1.0, -1.0)
            wconst = np.where(side % 2 == 1, 1.0, 0.0)
            mw = len(wI)
            M = m + mw
            if M == 0 or M > 6 * n + 8:
                break
            rows = np.arange(m)
            wr = m + np.arange(mw)
            idxM = n3 + np.arange(M)
            lam = None
            for it in range(iters):
                if time.time() > deadline:
                    break
                C = z[:n2].reshape(n, 2)
                R = z[n2:]
                dx = C[I, 0] - C[J, 0]
                dy = C[I, 1] - C[J, 1]
                g = np.empty(M)
                g[:m] = dx * dx + dy * dy - (R[I] + R[J]) ** 2
                g[m:] = wconst + wsgn * z[wcol] - R[wi]
                Jm = np.zeros((M, n3))
                Jm[rows, 2 * I] = 2 * dx
                Jm[rows, 2 * J] = -2 * dx
                Jm[rows, 2 * I + 1] = 2 * dy
                Jm[rows, 2 * J + 1] = -2 * dy
                s = -2 * (R[I] + R[J])
                Jm[rows, n2 + I] = s
                Jm[rows, n2 + J] = s
                Jm[wr, wcol] = wsgn
                Jm[wr, n2 + wi] = -1.0
                if lam is None:
                    G = Jm @ Jm.T
                    reg = 1e-10 * (1.0 + float(np.trace(G)) / M)
                    G[np.arange(M), np.arange(M)] += reg
                    try:
                        lam = np.linalg.solve(G, Jm @ gf)
                    except Exception:
                        lam = np.linalg.lstsq(Jm.T, gf, rcond=None)[0]
                    if not np.isfinite(lam).all():
                        lam = None
                        break
                stat = gf - Jm.T @ lam
                if it > 0 and abs(g).max() < 1e-15 and abs(stat).max() < 1e-12:
                    break
                W = np.zeros((n3, n3))
                if m:
                    lp = 2.0 * lam[:m]
                    for a, b, sg in (
                        (2 * I, 2 * I, -1.0),
                        (2 * J, 2 * J, -1.0),
                        (2 * I, 2 * J, 1.0),
                        (2 * J, 2 * I, 1.0),
                        (2 * I + 1, 2 * I + 1, -1.0),
                        (2 * J + 1, 2 * J + 1, -1.0),
                        (2 * I + 1, 2 * J + 1, 1.0),
                        (2 * J + 1, 2 * I + 1, 1.0),
                        (n2 + I, n2 + I, 1.0),
                        (n2 + J, n2 + J, 1.0),
                        (n2 + I, n2 + J, 1.0),
                        (n2 + J, n2 + I, 1.0),
                    ):
                        np.add.at(W, (a, b), sg * lp)
                N = n3 + M
                K = np.zeros((N, N))
                K[:n3, :n3] = W
                K[idx3, idx3] += rho
                K[:n3, n3:] = -Jm.T
                K[n3:, :n3] = Jm
                K[idxM, idxM] = -delta
                rhs = np.concatenate([-stat, -g])
                sol = None
                try:
                    sol = np.linalg.solve(K, rhs)
                    if not np.isfinite(sol).all() or abs(K @ sol - rhs).max() > 1e-8 * (1.0 + abs(rhs).max()):
                        sol = None
                except Exception:
                    sol = None
                if sol is None:
                    try:
                        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
                    except Exception:
                        break
                    if not np.isfinite(sol).all():
                        break
                p = sol[:n3]
                dl = sol[n3:]
                pm = float(abs(p).max())
                cap = 0.5 * rbar
                alpha = 1.0 if pm <= cap else cap / pm
                z = z + alpha * p
                lam = lam + alpha * dl
                z[:n2] = np.clip(z[:n2], 0.0, 1.0)
                z[n2:] = np.clip(z[n2:], 1e-9, 0.5)
                if alpha * pm < 1e-15:
                    break
            C2 = z[:n2].reshape(n, 2).copy()
            R2 = z[n2:].copy()
            for rr in (R2, lp_radii(C2, R2)):
                cand = evaluate(C2, rr)
                if cand["s"] > best["s"]:
                    best = cand
            gap2 = np.sqrt(((C2[I0] - C2[J0]) ** 2).sum(1)) - R2[I0] - R2[J0]
            bad = (gap2 < -1e-11) & ~act
            wg2 = np.concatenate([C2[:, 0] - R2, 1 - C2[:, 0] - R2, C2[:, 1] - R2, 1 - C2[:, 1] - R2])
            wbad = (wg2 < -1e-11) & ~wact
            neg = np.zeros(M, bool) if lam is None else (lam < -1e-7)
            if not bad.any() and not wbad.any() and not neg.any():
                break
            if neg.any():
                ai = np.nonzero(act)[0]
                act[ai[neg[:m]]] = False
                wa = np.nonzero(wact)[0]
                wact[wa[neg[m:]]] = False
            act |= bad
            wact |= wbad
    except Exception:
        pass
    return best


# ----------------------------------------------------------------------------- SLSQP active-set polish
def polish(c, r, n, deadline, rounds=4, maxiter=120):
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

    for _ in range(rounds):
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
                options={"maxiter": maxiter, "ftol": 1e-14},
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


def full_polish(sol, n, deadline, budget, slsqp=True, rounds=4, maxiter=120):
    """SLSQP (optional) followed by the exact Newton finish; returns the best strictly feasible result."""
    best = sol
    if best["s"] <= 0:
        return best
    if slsqp and time.time() < deadline:
        q = polish(best["c"], best["r"], n, deadline, rounds=rounds, maxiter=maxiter)
        if q["s"] > best["s"]:
            best = q
    if time.time() < deadline:
        q = newton_polish(best["c"], best["r"], n, min(deadline, time.time() + 0.04 * budget))
        if q["s"] > best["s"]:
            best = q
    return best


# ----------------------------------------------------------------------------- exact hole finding
def _hole_val(P, ck, rk):
    """Radius of the largest empty circle centred at each point of P (negative if the point is covered)."""
    wall = np.minimum(np.minimum(P[:, 0], 1 - P[:, 0]), np.minimum(P[:, 1], 1 - P[:, 1]))
    if len(ck):
        d = np.sqrt(((P[:, None, :] - ck[None, :, :]) ** 2).sum(-1)) - rk[None, :]
        return np.minimum(d.min(1), wall)
    return wall


def _circumcentres(T):
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    bx = b - a
    cx = c - a
    d = 2.0 * (bx[:, 0] * cx[:, 1] - bx[:, 1] * cx[:, 0])
    ok = np.abs(d) > 1e-12
    d = np.where(ok, d, 1.0)
    b2 = (bx * bx).sum(1)
    c2 = (cx * cx).sum(1)
    ux = (cx[:, 1] * b2 - bx[:, 1] * c2) / d
    uy = (bx[:, 0] * c2 - cx[:, 0] * b2) / d
    return (a + np.stack([ux, uy], 1))[ok]


def hole_candidates(ck, rk):
    """Delaunay circumcentres of the centres augmented with their wall mirror images (so wall-touching holes get
    candidates too). Empty array on any failure."""
    if Delaunay is None or len(ck) < 4:
        return np.zeros((0, 2))
    try:
        rbar = max(float(rk.mean()), 1e-6)
        band = 3.0 * rbar
        pts = [ck]
        for col, lo in ((0, True), (0, False), (1, True), (1, False)):
            sel = ck[:, col] < band if lo else ck[:, col] > 1 - band
            if sel.any():
                m = ck[sel].copy()
                m[:, col] = -m[:, col] if lo else 2.0 - m[:, col]
                pts.append(m)
        pts = np.concatenate(pts)
        if len(pts) > 4000:
            return np.zeros((0, 2))
        tri = Delaunay(pts)
        cc = _circumcentres(pts[tri.simplices])
        m = (cc[:, 0] > 0) & (cc[:, 0] < 1) & (cc[:, 1] > 0) & (cc[:, 1] < 1) & np.isfinite(cc).all(1)
        return cc[m]
    except Exception:
        return np.zeros((0, 2))


def _refine_hole(p, h0, ck, rk):
    """Nelder-Mead maximisation of the empty-circle radius from a candidate centre."""
    try:
        s = max(0.5 * float(h0), 1e-3)
        simplex = np.array([p, p + [s, 0.0], p + [0.0, s]])
        f = lambda q: -float(_hole_val(np.clip(q, 0.0, 1.0)[None, :], ck, rk)[0])
        res = minimize(
            f, p, method="Nelder-Mead", options={"maxfev": 80, "xatol": 1e-7, "fatol": 1e-10, "initial_simplex": simplex}
        )
        q = np.clip(res.x, 0.0, 1.0)
        return q, -float(res.fun)
    except Exception:
        return p, float(h0)


def fill_holes(ck, rk, k, rng, samples=3500, refine=True):
    """Greedy sequential placement of k circles into the largest empty holes: candidates are random samples plus
    Delaunay circumcentres (wall-mirrored), the best few are refined by Nelder-Mead to the exact local hole."""
    ck = np.asarray(ck, float).reshape(-1, 2)
    rk = np.asarray(rk, float).reshape(-1)
    P = rng.random((samples, 2))
    if refine:
        cc = hole_candidates(ck, rk)
        if len(cc):
            P = np.concatenate([P, cc])
    h = _hole_val(P, ck, rk)
    ck2 = ck.copy()
    rk2 = rk.copy()
    cs = np.zeros((k, 2))
    rs = np.zeros(k)
    for t in range(k):
        if refine:
            order = np.argsort(-h)[:3]
            bp = P[order[0]].copy()
            bh = float(h[order[0]])
            for j in order:
                q, hq = _refine_hole(P[j], h[j], ck2, rk2)
                if hq > bh and np.isfinite(q).all():
                    bp, bh = q, hq
        else:
            j = int(np.argmax(h))
            bp = P[j].copy()
            bh = float(h[j])
        bp = np.clip(bp, 1e-3, 1 - 1e-3)
        hr = max(bh, 1e-3)
        cs[t] = bp
        rs[t] = hr
        ck2 = np.concatenate([ck2, bp[None, :]])
        rk2 = np.concatenate([rk2, [hr]])
        h = np.minimum(h, np.sqrt(((P - bp) ** 2).sum(1)) - hr)
    return cs, rs


# ----------------------------------------------------------------------------- init & moves
def hex_lattice(n, rng):
    s3 = math.sqrt(3.0)
    full, part = [], []
    base = max(2, int(math.sqrt(n)))
    lo = n - max(2, int(0.15 * n))
    for nc in range(max(2, base - 3), base + 5):
        for variant in (0, 1):
            wunits = nc if variant == 0 else nc + 0.5
            r = 0.5 / wunits
            nr = int((1 - 2 * r) / (s3 * r) + 1e-9) + 1
            for rows in (nr, nr - 1):
                if rows < 1:
                    continue
                cnt = sum(nc - (i % 2) for i in range(rows)) if variant == 0 else nc * rows
                if cnt >= n:
                    if rows == nr:
                        full.append((r, nc, rows, variant))
                elif cnt >= lo:
                    part.append((r, nc, rows, variant))
    if not full and not part:
        return None
    use_part = bool(part) and (not full or rng.random() < 0.4)
    cands = part if use_part else full
    cands.sort(key=lambda t: -t[0])
    r, nc, nr, variant = cands[int(min(rng.integers(0, 3), len(cands) - 1))]
    pts = []
    for i in range(nr):
        y = r + i * s3 * r
        m = nc - (i % 2) if variant == 0 else nc
        x0 = r + (i % 2) * r
        for j in range(m):
            pts.append((x0 + 2 * r * j, y))
    pts = np.array(pts)
    slack = 1 - (pts[:, 1].max() + r)
    rad = np.full(len(pts), r)
    if use_part:
        if rng.random() < 0.5:
            pts[:, 1] += slack
        extra = n - len(pts)
        cs, rs = fill_holes(pts, rad, extra, rng, samples=4000)
        pts = np.concatenate([pts, cs])
        rad = np.concatenate([rad, 0.9 * rs])
    else:
        pts[:, 1] += 0.5 * slack
        extra = len(pts) - n
        if extra > 0:
            if rng.random() < 0.5:
                keep = rng.permutation(len(pts))[:n]
            else:
                th = rng.uniform(0, 2 * math.pi)
                key = pts @ np.array([math.cos(th), math.sin(th)]) + rng.normal(0, 0.3 * r, len(pts))
                keep = np.argsort(key)[:n]
            pts = pts[keep]
            rad = rad[keep]
    if len(pts) != n:
        return None
    if rng.random() < 0.5:
        pts = pts[:, ::-1].copy()
    c = np.clip(pts + rng.normal(0, 0.1 * r, pts.shape), 0.02, 0.98)
    rad = rad * rng.uniform(0.85, 1.0, n)
    return c, rad


def corner_lattice(n, rng):
    """Corner-anchored template: place slightly larger discs in the 4 corners then fill interior with a hex lattice inset.
    Exploits the fact that optimal packings often have larger circles touching two walls."""
    if n < 8:
        return None
    s3 = math.sqrt(3.0)
    k = int(math.ceil(math.sqrt(n)))
    r0 = 0.5 / k
    # corner circles: place at (r_c, r_c) with r_c ~ 1.15*r0
    rc = r0 * rng.uniform(1.05, 1.25)
    rc = min(rc, 0.18)
    corners = np.array([[rc, rc], [1 - rc, rc], [rc, 1 - rc], [1 - rc, 1 - rc]])
    crad = np.full(4, rc * rng.uniform(0.92, 1.0))
    # interior hex lattice inset by 2*rc
    inset = 2.2 * rc
    avail = 1 - 2 * inset
    if avail <= 0:
        return None
    # number of cols that fit
    nc = max(2, int(avail / (2 * r0)))
    nr = max(2, int(avail / (s3 * r0)))
    pts = []
    for i in range(nr):
        y = inset + r0 + i * s3 * r0
        if y + r0 > 1 - inset + 1e-9:
            break
        x0 = inset + r0 + (i % 2) * r0
        for j in range(nc - (i % 2)):
            x = x0 + 2 * r0 * j
            if x + r0 > 1 - inset + 1e-9:
                break
            pts.append((x, y))
    if not pts:
        return None
    pts = np.array(pts)
    need = n - 4
    if len(pts) < need:
        extra_pts, _ = fill_holes(np.concatenate([corners, pts]), np.concatenate([crad, np.full(len(pts), r0)]), need - len(pts), rng, samples=3000)
        pts = np.concatenate([pts, extra_pts])
    elif len(pts) > need:
        # wall-biased keep: prefer interior points farther from corners
        keep = rng.permutation(len(pts))[:need] if rng.random() < 0.5 else np.argsort(np.sqrt(((pts - 0.5)**2).sum(1)))[ :need]
        # bias: keep more central? actually we want diverse; use random 70%
        if rng.random() < 0.7:
            keep = rng.permutation(len(pts))[:need]
        pts = pts[keep]
    all_pts = np.concatenate([corners, pts])
    all_rad = np.concatenate([crad, np.full(len(pts), r0)])
    # small jitter and variable radii
    all_pts = np.clip(all_pts + rng.normal(0, 0.07 * r0, all_pts.shape), 0.02, 0.98)
    all_rad = all_rad * rng.uniform(0.82, 1.02, n)
    # ensure corner circles still near walls
    for i in range(4):
        all_pts[i] = np.clip(all_pts[i], rc * 0.7, 1 - rc * 0.7)
    return all_pts, all_rad


def _lattice_pts(ua, ub, s, off, cap=6000):
    a = s * ua
    b = s * ub
    dmin = s * min(
        math.hypot(ua[0], ua[1]),
        math.hypot(ub[0], ub[1]),
        math.hypot(ua[0] - ub[0], ua[1] - ub[1]),
        math.hypot(ua[0] + ub[0], ua[1] + ub[1]),
    )
    r = 0.5 * dmin
    if r >= 0.5 or r <= 1e-4:
        return np.zeros((0, 2)), r
    M = np.array([a, b]).T
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-12:
        return None, r
    Minv = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) - off
    ij = corners @ Minv.T
    i0 = int(math.floor(ij[:, 0].min())) - 1
    i1 = int(math.ceil(ij[:, 0].max())) + 1
    j0 = int(math.floor(ij[:, 1].min())) - 1
    j1 = int(math.ceil(ij[:, 1].max())) + 1
    if (i1 - i0 + 1) * (j1 - j0 + 1) > cap:
        return None, r
    I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    P = off + I.ravel()[:, None] * a + J.ravel()[:, None] * b
    m = (P[:, 0] >= r) & (P[:, 0] <= 1 - r) & (P[:, 1] >= r) & (P[:, 1] <= 1 - r)
    return P[m], r


def build_lattice_bank(n, rng, samples, deadline, keep=48):
    bank = []
    lo_cnt = max(2, n - int(0.15 * n))
    for _ in range(samples):
        if time.time() > deadline:
            break
        try:
            # mix hex and square lattices
            if rng.random() < 0.18:
                # square lattice
                th = rng.uniform(0, math.pi / 2)
                phi = math.pi / 2 + rng.normal(0, 0.06)
                ratio = 1.0 + rng.normal(0, 0.03)
            else:
                if rng.random() < 0.3:
                    th = (0.0 if rng.random() < 0.5 else math.pi / 6) + rng.normal(0, 0.01)
                else:
                    th = rng.uniform(0, math.pi / 3)
                phi = math.pi / 3 + rng.normal(0, 0.08)
                ratio = 1.0 + rng.normal(0, 0.04)
            ua = np.array([math.cos(th), math.sin(th)])
            ub = ratio * np.array([math.cos(th + phi), math.sin(th + phi)])
            off = rng.random(2)
            if rng.random() < 0.7:
                target = n
            else:
                target = n - int(rng.integers(1, max(2, int(0.12 * n)) + 1))
            det = abs(ua[0] * ub[1] - ua[1] * ub[0])
            if det < 0.3:
                continue
            s0 = math.sqrt(1.0 / (target * det))
            lo, hi = 0.7 * s0, 1.3 * s0
            ok = False
            for _k in range(8):
                P, r = _lattice_pts(ua, ub, lo, off)
                if P is not None and len(P) >= target:
                    ok = True
                    break
                lo *= 0.8
            if not ok:
                continue
            for _k in range(8):
                P, r = _lattice_pts(ua, ub, hi, off)
                if P is None or len(P) < target:
                    break
                hi *= 1.25
            for _k in range(24):
                mid = 0.5 * (lo + hi)
                P, r = _lattice_pts(ua, ub, mid, off)
                if P is not None and len(P) >= target:
                    lo = mid
                else:
                    hi = mid
            P, r = _lattice_pts(ua, ub, lo, off)
            if P is None:
                continue
            cnt = len(P)
            if cnt < lo_cnt or cnt > n + max(3, int(0.3 * n)):
                continue
            if cnt >= 2:
                D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
                np.fill_diagonal(D, np.inf)
                r = min(r, 0.5 * float(D.min()))
            if r <= 1e-4:
                continue
            score = r * min(cnt, n) + 0.35 * r * max(0, n - cnt)
            # bonus for square lattice at near-square n
            if abs(phi - math.pi/2) < 0.15:
                score *= 1.02
            bank.append((score, P.copy(), r))
        except Exception:
            continue
    bank.sort(key=lambda t: -t[0])
    return bank[:keep]


def affine_init(n, rng, bank):
    if not bank:
        return None
    idx = min(int(rng.exponential(5.0)), len(bank) - 1)
    _, P, r = bank[idx]
    P = P.copy()
    cnt = len(P)
    rad = np.full(cnt, r)
    if cnt > n:
        if rng.random() < 0.5:
            keep = rng.permutation(cnt)[:n]
        else:
            wall = np.minimum(np.minimum(P[:, 0], 1 - P[:, 0]), np.minimum(P[:, 1], 1 - P[:, 1]))
            key = wall + rng.normal(0, 0.7 * r, cnt)
            keep = np.argsort(-key)[:n]
        P = P[keep]
        rad = rad[keep]
    elif cnt < n:
        cs, rs = fill_holes(P, rad, n - cnt, rng, samples=4000)
        P = np.concatenate([P, cs])
        rad = np.concatenate([rad, 0.9 * rs])
    if len(P) != n:
        return None
    c = np.clip(P + rng.normal(0, 0.1 * r, P.shape), 0.02, 0.98)
    rad = rad * rng.uniform(0.85, 1.0, n)
    return c, rad


def init(n, rng, ictx=None):
    u = rng.random()
    if n >= 6 and u < 0.75:
        h = None
        try:
            if u < 0.12 and n >= 8:
                h = corner_lattice(n, rng)
            elif u < 0.30 or ictx is None:
                h = hex_lattice(n, rng)
            else:
                if ictx.get("bank") is None:
                    ictx["bank"] = build_lattice_bank(n, rng, ictx.get("samples", 140), ictx.get("bank_deadline", time.time() + 2.0))
                h = affine_init(n, rng, ictx["bank"])
                if h is None:
                    h = hex_lattice(n, rng)
        except Exception:
            h = None
        if h is not None:
            return h
        u = 0.9
    if u < 0.85:
        c = rng.random((n, 2))
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
    cs, rs = fill_holes(c[keep], r[keep], k, rng)
    c2, r2 = c.copy(), r.copy()
    c2[idx] = cs
    r2[idx] = rs
    return c2, r2


def _clip(c):
    return np.clip(c, 1e-3, 1 - 1e-3)


def sym_apply(c, k):
    """One of the 8 symmetries of the unit square (k = 1..7; 0 is the identity)."""
    x = c[:, 0].copy()
    y = c[:, 1].copy()
    if k & 1:
        x = 1.0 - x
    if k & 2:
        y = 1.0 - y
    if k & 4:
        x, y = y, x
    return np.stack([x, y], 1)


def lattice_dir(c, r, rng):
    """Dominant hex-lattice row orientation of a packing (circular mean of 6x contact angles), in (-pi/6, pi/6]."""
    try:
        rbar = max(float(r.mean()), 1e-6)
        I, J = build_pairs(c, r, 0.35 * rbar)
        if len(I) < 3:
            return rng.uniform(0, math.pi / 3)
        dx = c[J, 0] - c[I, 0]
        dy = c[J, 1] - c[I, 1]
        ang = np.arctan2(dy, dx)
        zc = np.exp(6j * ang).sum()
        if abs(zc) < 1e-9:
            return rng.uniform(0, math.pi / 3)
        return float(np.angle(zc)) / 6.0
    except Exception:
        return rng.uniform(0, math.pi / 3)


def contact_counts(c, r):
    n = len(c)
    rbar = max(float(r.mean()), 1e-6)
    I, J = build_pairs(c, r, 0.05 * rbar)
    cnt = np.bincount(I, minlength=n) + np.bincount(J, minlength=n)
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1])) - r
    cnt = cnt + (wall < 0.05 * rbar)
    return cnt


def pick_defect(c, r, rng, k, last, rbar):
    """Which circle(s) to take out: weakest (smallest), loosest (fewest contacts), a neighbour of the hole that was
    just filled (so the vacancy keeps walking), or random."""
    n = len(c)
    u = rng.random()
    if last is not None and u < 0.4 and n > 8:
        d = np.sqrt(((c - c[last]) ** 2).sum(1))
        d[last] = np.inf
        nb = np.argsort(d)[:6]
        return rng.choice(nb, k, replace=False)
    if u < 0.7:
        return np.argsort(r + rng.normal(0, 0.15 * rbar, n))[:k]
    if u < 0.87:
        try:
            cnt = contact_counts(c, r)
            return np.argsort(cnt + rng.random(n))[:k]
        except Exception:
            pass
    return rng.choice(n, k, replace=False)


def op_jitter(scale):
    def f(c, r, rng, n, ctx):
        rbar = max(float(r.mean()), 1e-6)
        c2 = c + rng.normal(0, rbar * scale, c.shape)
        return _clip(c2), r.copy(), (1e3, 1e4, 1e5)

    return f


def op_subset(c, r, rng, n, ctx):
    rbar = max(float(r.mean()), 1e-6)
    k = max(1, int(n * rng.uniform(0.1, 0.3)))
    idx = rng.choice(n, k, replace=False)
    c2 = c.copy()
    c2[idx] += rng.normal(0, rbar * 0.8, (k, 2))
    return _clip(c2), r.copy(), (1e2, 1e3, 1e4, 1e5)


def op_relocate(kmax):
    def f(c, r, rng, n, ctx):
        k = int(rng.integers(1, min(kmax, n) + 1))
        c2, r2 = relocate(c, r, rng, k)
        return _clip(c2), r2, (1e2, 1e3, 1e4, 1e5)

    return f


def op_swap(c, r, rng, n, ctx):
    order = np.argsort(r)
    q = max(1, n // 3)
    i = order[rng.integers(0, q)]
    j = order[n - 1 - rng.integers(0, q)]
    c2 = c.copy()
    c2[[i, j]] = c2[[j, i]]
    return _clip(c2), r.copy(), (1e2, 1e3, 1e4, 1e5)


def op_bignudge(c, r, rng, n, ctx):
    rbar = max(float(r.mean()), 1e-6)
    order = np.argsort(r)
    q = max(1, n // 3)
    i = order[n - 1 - rng.integers(0, q)]
    th = rng.uniform(0, 2 * math.pi)
    c2 = c.copy()
    c2[i] += rbar * rng.uniform(0.6, 1.5) * np.array([math.cos(th), math.sin(th)])
    return _clip(c2), r.copy(), (1e2, 1e3, 1e4, 1e5)


def op_reinsert(c, r, rng, n, ctx):
    rbar = max(float(r.mean()), 1e-6)
    k = int(rng.integers(1, min(3, n) + 1))
    idx = rng.choice(n, k, replace=False)
    c2, r2 = c.copy(), r.copy()
    c2[idx] = rng.random((k, 2))
    r2[idx] = 0.3 * rbar
    return _clip(c2), r2, (1e2, 1e3, 1e4, 1e5)


def op_local(c, r, rng, n, ctx):
    rbar = max(float(r.mean()), 1e-6)
    i = rng.integers(0, n)
    d = np.sqrt(((c - c[i]) ** 2).sum(1))
    idx = np.nonzero(d < 3.0 * rbar)[0]
    c2 = c.copy()
    c2[idx] += rng.normal(0, rbar * 0.5, (len(idx), 2))
    return _clip(c2), r.copy(), (1e2, 1e3, 1e4, 1e5)


def op_grain(c, r, rng, n, ctx):
    """Rotate the circles inside a disc (a local grain) by a sizeable angle about a circle centre or a hole: creates a
    misoriented grain / grain boundary that the relaxation then heals into a different defect arrangement."""
    rbar = max(float(r.mean()), 1e-6)
    if rng.random() < 0.5:
        p = c[int(rng.integers(0, n))].copy()
    else:
        p = rng.random(2)
    R = rbar * rng.uniform(2.0, 4.0)
    ang = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(math.pi / 9, math.pi / 3)
    d = np.sqrt(((c - p) ** 2).sum(1))
    idx = np.nonzero(d < R)[0]
    if len(idx) < 2:
        return op_local(c, r, rng, n, ctx)
    ca, sa = math.cos(ang), math.sin(ang)
    v = c[idx] - p
    rot = np.stack([ca * v[:, 0] - sa * v[:, 1], sa * v[:, 0] + ca * v[:, 1]], 1)
    c2 = c.copy()
    c2[idx] = p + rot
    rr = np.minimum(r, 0.5)[:, None]
    c2 = np.minimum(np.maximum(c2, rr), 1 - rr)
    return _clip(c2), r.copy(), (1e2, 1e3, 1e4, 1e5)


def op_slip(mode):
    """Lattice slip: shift one hex row (mode 1) or a band of rows / a half-plane (mode 2) along the row direction by
    about half a lattice period. Circles pushed past a wall are clipped to the wall or re-inserted into holes."""

    def f(c, r, rng, n, ctx):
        rbar = max(float(r.mean()), 1e-6)
        if rng.random() < 0.25:
            th = 0.0 if rng.random() < 0.5 else math.pi / 2
        else:
            th = lattice_dir(c, r, rng) + int(rng.integers(0, 3)) * math.pi / 3
        u = np.array([math.cos(th), math.sin(th)])
        nv = np.array([-u[1], u[0]])
        t = c @ nv
        i = int(rng.integers(0, n))
        h = math.sqrt(3.0) * rbar
        k = 1 if mode == 1 else int(rng.integers(2, 4))
        if mode == 2 and rng.random() < 0.3:
            band = t > t[i] - 0.5 * h
        else:
            band = (t > t[i] - 0.5 * h) & (t < t[i] + (k - 0.5) * h)
        if not band.any():
            band[i] = True
        mag = 1.0 if rng.random() < 0.6 else rng.uniform(0.5, 1.5)
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        c2 = c.copy()
        r2 = r.copy()
        c2[band] += sgn * mag * rbar * u
        rr = np.minimum(r2, 0.5)
        out = (c2[:, 0] < rr) | (c2[:, 0] > 1 - rr) | (c2[:, 1] < rr) | (c2[:, 1] > 1 - rr)
        if out.any():
            if rng.random() < 0.5:
                c2 = np.minimum(np.maximum(c2, rr[:, None]), 1 - rr[:, None])
            else:
                idx = np.nonzero(out)[0]
                keep = ~out
                cs, rs = fill_holes(c2[keep], r2[keep], len(idx), rng)
                c2[idx] = cs
                r2[idx] = rs
        return _clip(c2), r2, (10, 1e2, 1e3, 1e4, 1e5)

    return f


def op_cross(c, r, rng, n, ctx):
    pool = ctx["pool"]
    partners = [p for p in pool if abs(p["s"] - ctx["cur_s"]) > 1e-9 and len(p["c"]) == n]
    if not partners:
        return op_subset(c, r, rng, n, ctx)
    p = partners[int(rng.integers(0, len(partners)))]
    pc = p["c"]
    if rng.random() < 0.5:
        pc = sym_apply(pc, int(rng.integers(1, 8)))
    th = rng.uniform(0, math.pi)
    nv = np.array([math.cos(th), math.sin(th)])
    q = rng.uniform(0.25, 0.75, 2)
    sa = (c - q) @ nv
    sb = (pc - q) @ nv
    A = sa <= 0
    B = sb > 0
    cA, rA = c[A], r[A]
    cB, rB = pc[B], p["r"][B]
    if len(cA) and len(cB):
        d = np.sqrt(((cB[:, None, :] - cA[None, :, :]) ** 2).sum(-1)) - rA[None, :] - rB[:, None]
        ok = d.min(1) > -0.5 * rB
        cB, rB = cB[ok], rB[ok]
    c2 = np.concatenate([cA, cB]) if len(cB) else cA.copy()
    r2 = np.concatenate([rA, rB]) if len(rB) else rA.copy()
    m = len(c2)
    if m > n:
        keep = np.argsort(-r2)[:n]
        c2, r2 = c2[keep], r2[keep]
    elif m < n:
        cs, rs = fill_holes(c2, r2, n - m, rng)
        c2 = np.concatenate([c2, cs]) if m else cs
        r2 = np.concatenate([r2, rs]) if m else rs
    return _clip(c2), r2, (1e2, 1e3, 1e4, 1e5)


def op_migrate(mode):
    """Defect migration with an intermediate relaxation.
    mode 0 (vacancy diffusion chain): take out the weakest / loosest / a neighbour of the last refilled hole, let the
      remaining packing re-equilibrate (the vacancy smears out and the largest hole moves), re-insert into the exact
      largest hole(s); repeated 1-3 times so the defect random-walks away from where it started.
    mode 1 (insert-relax-evict): push an extra circle into the largest hole, relax the n+1 packing under that
      pressure, then evict the weakest ORIGINAL circle, so a defect migrates from the weakest site to the hole.
    Plain remove-and-refill (without relaxation) refills the very hole the circle just left and lands back in the
    same basin; the intermediate relaxation is what makes these moves structural."""

    def f(c, r, rng, n, ctx):
        if n < 4:
            return op_subset(c, r, rng, n, ctx)
        dl = ctx.get("deadline", time.time() + 1.0)
        rbar = max(float(r.mean()), 1e-6)
        if mode == 0:
            steps = 1 + int(rng.random() < 0.5) + int(rng.random() < 0.3)
            c2, r2 = c.copy(), r.copy()
            last = None
            for s in range(steps):
                if time.time() > dl:
                    break
                k = int(rng.integers(1, 3)) if s == 0 else 1
                idx = np.asarray(pick_defect(c2, r2, rng, k, last, rbar))
                keep = np.ones(n, bool)
                keep[idx] = False
                ck, rk = run_penalty(c2[keep], r2[keep], n - k, (1e3, 1e4), 60, dl)
                cs, rs = fill_holes(ck, rk, k, rng, samples=2500)
                c2[keep] = ck
                r2[keep] = rk
                c2[idx] = cs
                r2[idx] = 0.9 * rs
                last = int(idx[0])
            return _clip(c2), r2, (1e3, 1e4, 1e5)
        cs, rs = fill_holes(c, r, 1, rng, samples=2500)
        rnew = max(0.9 * float(rs[0]), rng.uniform(0.4, 0.8) * rbar)
        c1 = np.concatenate([c, cs])
        r1 = np.concatenate([r, [rnew]])
        c1, r1 = run_penalty(c1, r1, n + 1, (1e2, 1e3, 1e4), 60, dl)
        u = rng.random()
        if u < 0.6:
            j = int(np.argmin(r1[:n] + rng.normal(0, 0.15 * rbar, n)))
        elif u < 0.8:
            try:
                cnt = contact_counts(c1[:n], r1[:n])
                j = int(np.argmin(cnt + rng.random(n)))
            except Exception:
                j = int(np.argmin(r1[:n]))
        else:
            j = int(np.argmin(r1 + rng.normal(0, 0.15 * rbar, n + 1)))
        c2 = c1[:n].copy()
        r2 = r1[:n].copy()
        if j < n:
            c2[j] = c1[n]
            r2[j] = r1[n]
        return _clip(c2), r2, (1e3, 1e4, 1e5)

    return f


def op_rescale(c, r, rng, n, ctx):
    """Anneal radii: stretch large circles slightly and shrink small ones (or vice versa) then re-relax.
    Explores the variable-radii dimension which L-BFGS-B on positions alone cannot."""
    rbar = max(float(r.mean()), 1e-6)
    order = np.argsort(r)
    fac = rng.uniform(0.92, 1.08, n)
    # bias: largest +3%, smallest -3%
    q = max(1, n // 4)
    fac[order[-q:]] *= rng.uniform(1.02, 1.06)
    fac[order[:q]] *= rng.uniform(0.94, 0.98)
    r2 = np.clip(r * fac, 1e-4, 0.5)
    # clip to wall distance
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), np.minimum(c[:, 1], 1 - c[:, 1]))
    r2 = np.minimum(r2, wall - 1e-4)
    r2 = np.maximum(r2, 1e-4)
    c2 = c + rng.normal(0, 0.06 * rbar, c.shape)
    return _clip(c2), r2, (1e2, 1e3, 1e4, 1e5)


def op_boundary(c, r, rng, n, ctx):
    """Push boundary circles inward/outward: perturb wall-touching circles along the wall normal."""
    rbar = max(float(r.mean()), 1e-6)
    wall_gap = np.minimum(np.minimum(c[:, 0] - r, 1 - c[:, 0] - r), np.minimum(c[:, 1] - r, 1 - c[:, 1] - r))
    bidx = np.where(wall_gap < 0.10 * rbar)[0]
    if len(bidx) == 0:
        return op_jitter(0.08)(c, r, rng, n, ctx)
    k = min(len(bidx), max(1, int(rng.integers(1, 4))))
    sel = rng.choice(bidx, k, replace=False)
    c2 = c.copy()
    for i in sel:
        # push slightly away from nearest wall
        dx = 0.0
        dy = 0.0
        if c[i, 0] - r[i] < 0.08 * rbar:
            dx = rng.uniform(0.2, 0.8) * rbar
        elif 1 - c[i, 0] - r[i] < 0.08 * rbar:
            dx = -rng.uniform(0.2, 0.8) * rbar
        if c[i, 1] - r[i] < 0.08 * rbar:
            dy = rng.uniform(0.2, 0.8) * rbar
        elif 1 - c[i, 1] - r[i] < 0.08 * rbar:
            dy = -rng.uniform(0.2, 0.8) * rbar
        if dx == 0 and dy == 0:
            dx = rng.normal(0, 0.3 * rbar)
            dy = rng.normal(0, 0.3 * rbar)
        c2[i, 0] += dx
        c2[i, 1] += dy
    return _clip(c2), r.copy(), (1e2, 1e3, 1e4, 1e5)


OPS = [
    op_jitter(0.05),
    op_jitter(0.15),
    op_jitter(0.4),
    op_subset,
    op_relocate(2),
    op_relocate(4),
    op_swap,
    op_bignudge,
    op_reinsert,
    op_local,
    op_cross,
    op_slip(1),
    op_slip(2),
    op_migrate(0),
    op_migrate(1),
    op_grain,
    op_rescale,
    op_boundary,
]


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


def solve(n, budget, seed, out, t0=None, peers=()):
    rng = np.random.default_rng(seed)
    if t0 is None:
        t0 = time.time()
    reserve = min(5.0, max(0.5, 0.05 * budget))
    end = t0 + budget - reserve
    p1_end = t0 + 0.25 * budget
    p2_end = t0 + 0.82 * budget - reserve
    p4_end = end - max(0.3, 0.02 * budget)
    best = trivial(n)
    save(out, n, best)
    if n == 1:
        best = evaluate(np.array([[0.5, 0.5]]), np.array([0.5]))
        save(out, n, best)
        return best

    pool = []

    def add_pool(sol):
        if sol["s"] <= 0:
            return
        for p in pool:
            if abs(p["s"] - sol["s"]) < 1e-6:
                return
        pool.append(sol)
        pool.sort(key=lambda p: -p["s"])
        del pool[8:]

    exch_dt = max(1.0, 0.04 * budget)
    next_exch = [time.time() + exch_dt]

    def exchange(force=False):
        if not peers:
            return
        if not force and time.time() < next_exch[0]:
            return
        next_exch[0] = time.time() + exch_dt
        for path in peers:
            cand = load_candidate(path, n)
            if cand is not None:
                add_pool(cand)

    ictx = {
        "bank": None,
        "samples": int(min(400, max(80, 5 * budget))),
        "bank_deadline": min(t0 + max(0.6, 0.035 * budget), end),
    }

    # Phase 1: cold multi-start, each finished by the exact Newton polish
    starts = 0
    while time.time() < p1_end or starts < 2:
        if time.time() > end:
            break
        try:
            c, r = init(n, rng, ictx)
        except Exception:
            c, r = rng.random((n, 2)), np.full(n, 0.4 / math.sqrt(n))
        c, r = run_penalty(c, r, n, (10, 100, 1e3, 1e4, 1e5), 320, end)
        cand = evaluate(c, lp_radii(c, r))
        starts += 1
        if cand["s"] > 0 and time.time() < end:
            q = newton_polish(cand["c"], cand["r"], n, min(end, time.time() + 0.025 * budget))
            if q["s"] > cand["s"]:
                cand = q
        add_pool(cand)
        if cand["s"] > best["s"]:
            best = cand
            save(out, n, best)

    if time.time() < end and best["s"] > 0:
        cand = full_polish(best, n, min(end, time.time() + 0.14 * budget), budget)
        if cand["s"] > best["s"]:
            best = cand
            save(out, n, best)
    add_pool(best)
    exchange(True)

    # Phase 2 / 4: basin hopping on the incumbent with adaptive operator selection; Newton-exact comparison
    cur = best
    stall = 0
    last_polish = time.time()
    W = np.ones(len(OPS))
    nt_time = 0.0
    hop_t0 = time.time()

    def hop_phase(until, slsqp_ok):
        nonlocal best, cur, stall, last_polish, nt_time
        while time.time() < until and cur["s"] > 0:
            exchange()
            p = (0.1 + W) / (0.1 + W).sum()
            k = int(rng.choice(len(OPS), p=p))
            ctx = {"pool": pool, "cur_s": cur["s"], "deadline": until}
            try:
                c, r, mus = OPS[k](cur["c"], cur["r"], rng, n, ctx)
                if len(c) != n or len(r) != n:
                    continue
            except Exception:
                continue
            c, r = run_penalty(c, r, n, mus, 200, until)
            cand = evaluate(c, lp_radii(c, r))
            now = time.time()
            if cand["s"] > 0 and now < until:
                rbar = max(float(cur["r"].mean()), 1e-6)
                same = len(cand["c"]) == len(cur["c"]) and float(np.abs(cand["c"] - cur["c"]).max()) < 0.02 * rbar
                near = cand["s"] > cur["s"] - 3e-3
                budget_ok = nt_time < 0.5 * (now - hop_t0) + 1.0
                if not same and (cand["s"] > cur["s"] - 5e-4 or (near and budget_ok)):
                    tq = time.time()
                    q = newton_polish(cand["c"], cand["r"], n, min(until, tq + 0.022 * budget))
                    nt_time += time.time() - tq
                    if q["s"] > cand["s"]:
                        cand = q
            prog = (time.time() - t0) / max(budget, 1e-9)
            T = 3e-4 * max(0.0, 1 - prog)
            reward = 0.0
            if cand["s"] > 0 and cand["s"] > cur["s"]:
                reward = 0.3
            if cand["s"] > 0 and cand["s"] > cur["s"] - T * rng.exponential():
                cur = cand
            add_pool(cand)
            if cand["s"] > best["s"]:
                reward = 1.0
                best = cand
                cur = best
                stall = 0
                save(out, n, best)
                if time.time() < until:
                    use_slsqp = slsqp_ok and (n <= 60 or time.time() - last_polish > 0.13 * budget)
                    if use_slsqp:
                        last_polish = time.time()
                    pol = full_polish(best, n, min(until, time.time() + 0.1 * budget), budget, slsqp=use_slsqp)
                    if pol["s"] > best["s"]:
                        best = pol
                        cur = best
                        save(out, n, best)
                        add_pool(best)
            else:
                stall += 1
                if stall > 30:
                    stall = 0
                    exchange(True)
                    if pool:
                        idx = min(int(rng.exponential(0.8)), len(pool) - 1)
                        cur = pool[idx]
                    else:
                        cur = best
            W[k] = 0.9 * W[k] + 0.1 * reward

    hop_phase(p2_end, True)

    # Phase 3: full polish of the best basin
    if time.time() < end and best["s"] > 0:
        pol = full_polish(best, n, min(end, time.time() + 0.12 * budget), budget)
        if pol["s"] > best["s"]:
            best = pol
            save(out, n, best)
            add_pool(best)

    # Phase 4: reclaim the remaining time with more (Newton-only) hops, then a final exact finish
    cur = best
    stall = 0
    try:
        hop_phase(p4_end, False)
    except Exception:
        pass
    if time.time() < end and best["s"] > 0:
        q = newton_polish(best["c"], best["r"], n, end)
        if q["s"] > best["s"]:
            best = q
            save(out, n, best)
    return best


# ----------------------------------------------------------------------------- island orchestration
def cpu_quota():
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().split()
        if parts and parts[0] != "max":
            return max(1, int(parts[0]) // int(parts[1]))
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read().strip())
        if q > 0 and p > 0:
            return max(1, q // p)
    except Exception:
        pass
    return None


def pick_workers(budget):
    env = os.environ.get("CSQV_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if budget < 10:
        return 1
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count() or 1
    q = cpu_quota()
    if q:
        ncpu = min(ncpu, q)
    try:
        ncpu = min(ncpu, int(ncpu - os.getloadavg()[0] + 0.5))
    except Exception:
        pass
    return max(1, min(ncpu, 8))


def worker_main(n, budget, seed, out, t0, peers):
    try:
        best = solve(n, budget, seed, out, t0=t0, peers=peers)
        save(out, n, best)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    n, budget, out = a.n, a.time, a.out
    reserve = min(5.0, max(0.5, 0.05 * budget))
    procs, peers = [], []
    nw = pick_workers(budget) if n > 1 else 1
    if nw > 1:
        try:
            import multiprocessing as mp

            methods = mp.get_all_start_methods()
            ctx = mp.get_context("fork") if "fork" in methods else mp.get_context()
            files = [out] + ["%s.w%d" % (out, w) for w in range(1, nw)]
            for w in range(1, nw):
                others = [f for f in files if f != files[w]]
                p = ctx.Process(
                    target=worker_main,
                    args=(n, budget, a.seed + 7919 * w, files[w], t0, others),
                    daemon=True,
                )
                p.start()
                procs.append(p)
                peers.append(files[w])
        except Exception:
            pass
    best = None
    try:
        best = solve(n, budget, a.seed, out, t0=t0, peers=peers)
    except Exception:
        best = None
    if best is None or best["s"] <= 0:
        best = load_candidate(out, n) or trivial(n)
    join_until = t0 + budget - 0.5 * reserve
    for p in procs:
        try:
            p.join(max(0.0, join_until - time.time()))
        except Exception:
            pass
    for path in peers:
        cand = load_candidate(path, n)
        if cand is not None and cand["s"] > best["s"]:
            best = cand
    save(out, n, best)
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(1.0)
        except Exception:
            pass
    for path in peers:
        for f in (path, path + ".tmp"):
            try:
                os.remove(f)
            except Exception:
                pass


if __name__ == "__main__":
    main()
```
