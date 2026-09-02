"""Island-model basin-hopping solver for csqv: hex-lattice template inits (incl. under-full lattice + strip fill),
sparse-neighbour penalty L-BFGS-B, adaptive move selection incl. half-plane crossover with elite pool, LP-optimal
radii for fixed centres, SLSQP active-set polish on the contact graph, final strict shrink. Independent chains run in
worker processes (one per available core) and exchange elites through atomic JSON files.
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
                I, J = build_pairs(C, R, 2.5 * rbar + 0.02)
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


# ----------------------------------------------------------------------------- init & moves
def fill_holes(ck, rk, k, rng, samples=3000):
    P = rng.random((samples, 2))
    wall = np.minimum(np.minimum(P[:, 0], 1 - P[:, 0]), np.minimum(P[:, 1], 1 - P[:, 1]))
    if len(ck):
        h = np.minimum((np.sqrt(((P[:, None, :] - ck[None, :, :]) ** 2).sum(-1)) - rk[None, :]).min(1), wall)
    else:
        h = wall
    cs = np.zeros((k, 2))
    rs = np.zeros(k)
    for t in range(k):
        j = int(np.argmax(h))
        hr = max(float(h[j]), 1e-3)
        cs[t] = P[j]
        rs[t] = hr
        h = np.minimum(h, np.sqrt(((P - P[j]) ** 2).sum(1)) - hr)
    return cs, rs


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
        # under-full lattice: leave the leftover strip on one side and fill it (and other holes) greedily
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


def init(n, rng):
    u = rng.random()
    if u < 0.5 and n >= 6:
        h = None
        try:
            h = hex_lattice(n, rng)
        except Exception:
            h = None
        if h is not None:
            return h
        u = 0.9
    if u < 0.7:
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


def op_cross(c, r, rng, n, ctx):
    # half-plane crossover: keep incumbent on one side of a random line, splice an elite's circles on the other side
    pool = ctx["pool"]
    partners = [p for p in pool if abs(p["s"] - ctx["cur_s"]) > 1e-9 and len(p["c"]) == n]
    if not partners:
        return op_subset(c, r, rng, n, ctx)
    p = partners[int(rng.integers(0, len(partners)))]
    th = rng.uniform(0, math.pi)
    nv = np.array([math.cos(th), math.sin(th)])
    q = rng.uniform(0.25, 0.75, 2)
    sa = (c - q) @ nv
    sb = (p["c"] - q) @ nv
    A = sa <= 0
    B = sb > 0
    cA, rA = c[A], r[A]
    cB, rB = p["c"][B], p["r"][B]
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
        # island model: pull sibling workers' current bests into the elite pool (crossover partners / restarts)
        if not peers:
            return
        if not force and time.time() < next_exch[0]:
            return
        next_exch[0] = time.time() + exch_dt
        for path in peers:
            cand = load_candidate(path, n)
            if cand is not None:
                add_pool(cand)

    # Phase 1: cold multi-start (hex / hex+strip / square / random templates)
    starts = 0
    while time.time() < p1_end or starts < 2:
        if time.time() > end:
            break
        try:
            c, r = init(n, rng)
        except Exception:
            c, r = rng.random((n, 2)), np.full(n, 0.4 / math.sqrt(n))
        c, r = run_penalty(c, r, n, (10, 100, 1e3, 1e4, 1e5), 300, end)
        cand = evaluate(c, lp_radii(c, r))
        starts += 1
        add_pool(cand)
        if cand["s"] > best["s"]:
            best = cand
            save(out, n, best)

    if time.time() < end and best["s"] > 0:
        cand = polish(best["c"], best["r"], n, min(end, time.time() + 0.12 * budget))
        if cand["s"] > best["s"]:
            best = cand
            save(out, n, best)
    add_pool(best)
    exchange(True)

    # Phase 2: basin hopping on the incumbent with adaptive operator selection
    cur = best
    stall = 0
    last_polish = time.time()
    W = np.ones(len(OPS))
    qp_time = 0.0
    p2_start = time.time()
    qmaxiter = 60 if n <= 60 else 40
    while time.time() < p2_end and cur["s"] > 0:
        exchange()
        p = (0.1 + W) / (0.1 + W).sum()
        k = int(rng.choice(len(OPS), p=p))
        ctx = {"pool": pool, "cur_s": cur["s"]}
        try:
            c, r, mus = OPS[k](cur["c"], cur["r"], rng, n, ctx)
            if len(c) != n or len(r) != n:
                continue
        except Exception:
            continue
        c, r = run_penalty(c, r, n, mus, 200, end)
        cand = evaluate(c, lp_radii(c, r))
        now = time.time()
        if (
            cand["s"] > 0
            and cand["s"] < best["s"]
            and cand["s"] > cur["s"] - 2e-3
            and qp_time < 0.3 * (now - p2_start) + 1.0
            and now < p2_end
        ):
            tq = time.time()
            q = polish(cand["c"], cand["r"], n, min(p2_end, tq + 0.03 * budget), rounds=2, maxiter=qmaxiter)
            qp_time += time.time() - tq
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
            if (n <= 60 or time.time() - last_polish > 0.15 * budget) and time.time() < p2_end:
                pol = polish(best["c"], best["r"], n, min(p2_end, time.time() + 0.1 * budget))
                last_polish = time.time()
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

    # Phase 3: final polish of the best basin
    if time.time() < end and best["s"] > 0:
        pol = polish(best["c"], best["r"], n, end)
        if pol["s"] > best["s"]:
            best = pol
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
