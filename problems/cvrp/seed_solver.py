"""Seed CVRP solver: Clarke-Wright parallel savings, then local search, within a wall-clock budget.

Phase 1: Clarke-Wright savings builds a feasible set of routes (each customer served once, capacity
respected) and it is written to disk immediately, so a timeout always leaves a feasible solution.
Phase 2: local search improves it until the deadline -- 2-opt inside each route, plus relocate and swap
moves between routes, both capacity-checked. Every improvement is saved atomically.

    python seed_solver.py --target X-n280-k17 --time 120 --seed 1 --out sol.json
writes {"target", "obj", "solution": {"routes": [[customer, ...], ...]}} (customer numbers 1..DIMENSION-1).
Pure python + numpy, no external solver.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone use; the loop also sets PYTHONPATH
from verify import dist_matrix, load_instance  # noqa: E402


def clarke_wright(D, demand, cap):
    """Parallel savings. Returns a list of routes (lists of customer node indices 1..n)."""
    n = len(demand) - 1
    routes = {i: [i] for i in range(1, n + 1)}  # route id == its initial customer
    where = {i: i for i in range(1, n + 1)}  # customer -> route id
    load = {i: float(demand[i]) for i in range(1, n + 1)}
    cust = np.arange(1, n + 1)
    i, j = np.meshgrid(cust, cust, indexing="ij")
    mask = i < j
    sav = (D[0, i] + D[0, j] - D[i, j])[mask]
    order = np.argsort(-sav, kind="stable")
    ii, jj = i[mask][order], j[mask][order]
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = where[a], where[b]
        if ra == rb or load[ra] + load[rb] > cap:
            continue
        Ra, Rb = routes[ra], routes[rb]
        if a not in (Ra[0], Ra[-1]) or b not in (Rb[0], Rb[-1]):
            continue
        if Ra[-1] != a:  # want a at the tail of Ra
            Ra.reverse()
        if Rb[0] != b:  # want b at the head of Rb
            Rb.reverse()
        merged = Ra + Rb
        for c in Rb:
            where[c] = ra
        routes[ra] = merged
        load[ra] += load[rb]
        del routes[rb]
    return list(routes.values())


def two_opt(route, D, deadline):
    """In-route 2-opt over depot(0)-...-depot(0); reverses segments while it lowers the route cost."""
    path = [0, *route, 0]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, len(path) - 2):
            a, b = path[i - 1], path[i]
            for k in range(i + 1, len(path) - 1):
                c, d = path[k], path[k + 1]
                if a == c:
                    continue
                if D[a, path[k]] + D[b, d] < D[a, b] + D[c, d]:
                    path[i : k + 1] = path[i : k + 1][::-1]
                    improved = True
                    a, b = path[i - 1], path[i]
    return path[1:-1]


def local_search(routes, D, demand, cap, deadline, save):
    """Relocate + swap between routes and 2-opt within routes until no improvement or the deadline."""
    load = [float(demand[r].sum()) for r in routes]

    def cost():
        return sum(_route_cost(r, D) for r in routes)

    best = cost()
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # relocate: move one customer to the best feasible position in another route
        for ai in range(len(routes)):
            if time.time() >= deadline:
                break
            A = routes[ai]
            for pos in range(len(A)):
                c = A[pos]
                prev = A[pos - 1] if pos > 0 else 0
                nxt = A[pos + 1] if pos + 1 < len(A) else 0
                gain = D[prev, c] + D[c, nxt] - D[prev, nxt]
                dc = float(demand[c])
                for bi in range(len(routes)):
                    if bi == ai:
                        continue
                    if load[bi] + dc > cap:
                        continue
                    B = routes[bi]
                    aug = [0, *B, 0]
                    bestpos, bestdelta = None, -1e-9
                    for u in range(len(aug) - 1):
                        add = D[aug[u], c] + D[c, aug[u + 1]] - D[aug[u], aug[u + 1]]
                        if add - gain < bestdelta:
                            bestdelta, bestpos = add - gain, u
                    if bestpos is not None:
                        A.pop(pos)
                        B.insert(bestpos, c)
                        load[ai] -= dc
                        load[bi] += dc
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
        if changed:
            routes[:] = [r for r in routes if r]
            load = [float(demand[r].sum()) for r in routes]
            cur = cost()
            if cur < best:
                best = cur
                save(routes, best)
            continue
        # swap: exchange two customers between different routes
        for ai in range(len(routes)):
            if time.time() >= deadline:
                break
            A = routes[ai]
            for pa in range(len(A)):
                ca = A[pa]
                pra, nxa = (A[pa - 1] if pa > 0 else 0), (A[pa + 1] if pa + 1 < len(A) else 0)
                for bi in range(ai + 1, len(routes)):
                    B = routes[bi]
                    for pb in range(len(B)):
                        cb = B[pb]
                        da, db = float(demand[ca]), float(demand[cb])
                        if load[ai] - da + db > cap or load[bi] - db + da > cap:
                            continue
                        prb, nxb = (B[pb - 1] if pb > 0 else 0), (B[pb + 1] if pb + 1 < len(B) else 0)
                        before = D[pra, ca] + D[ca, nxa] + D[prb, cb] + D[cb, nxb]
                        after = D[pra, cb] + D[cb, nxa] + D[prb, ca] + D[ca, nxb]
                        if after < before - 1e-9:
                            A[pa], B[pb] = cb, ca
                            load[ai] += db - da
                            load[bi] += da - db
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
        if changed:
            cur = cost()
            if cur < best:
                best = cur
                save(routes, best)
            continue
        # no inter-route move helped: polish each route with 2-opt, save if it lowered the total
        for idx in range(len(routes)):
            routes[idx] = two_opt(routes[idx], D, deadline)
        cur = cost()
        if cur < best - 1e-9:
            best = cur
            save(routes, best)
            changed = True
    return routes, best


def _route_cost(route, D):
    if not route:
        return 0
    path = [0, *route, 0]
    return int(sum(D[path[i], path[i + 1]] for i in range(len(path) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--time", type=float, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + max(2.0, a.time - 2.0)  # leave headroom before the loop's hard kill
    np.random.seed(a.seed % (2**31 - 1))

    inst = load_instance(a.target)
    D = dist_matrix(inst["coords"])
    demand, cap = inst["demand"], inst["capacity"]

    def save(routes, obj):
        d = {"target": a.target, "obj": int(obj), "solution": {"routes": [list(map(int, r)) for r in routes if r]}}
        tmp = a.out + ".tmp"
        json.dump(d, open(tmp, "w"))
        os.replace(tmp, a.out)

    routes = clarke_wright(D, demand, cap)
    routes = [two_opt(r, D, deadline) for r in routes]
    save(routes, sum(_route_cost(r, D) for r in routes))  # feasible on disk before local search
    local_search(routes, D, demand, cap, deadline, save)


if __name__ == "__main__":
    main()
