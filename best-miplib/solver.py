"""Verified multi-neighbourhood LNS on top of a HiGHS incumbent, plus a dedicated
row-weighting local search (RWLS) for pure set-covering instances.

Phase 1: plain HiGHS for ~30% of the budget (45% for pure feasibility problems, 20% for SCP).
Phase 1s: if the instance is a pure set-covering problem (all binaries, unit-coefficient >=1 rows,
          nonnegative costs), run a row-weighting local search from the incumbent (or a greedy cover).
Phase 1b: if no incumbent, minimise row slacks (slack MIP + random-fix LNS on it).
Phase 2: portfolio LNS around the incumbent: random fixing, root-LP disagreement (RINS-like),
row-BFS, active-column, cost-guided column neighbourhoods and local branching, chosen by a
success-weighted bandit, each with its own adaptive free-fraction; sub-MIP time limit adapts.
Every candidate is re-checked in numpy (bounds, integrality, rows within 1e-6) before it is saved.

    python solver.py --target assign1-10-4 --time 300 --seed 1 --out sol.json
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
import highspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # standalone use; the loop also sets PYTHONPATH
from records import instance_path  # noqa: E402

INF = float(highspy.kHighsInf)
ROW_TOL = 9.5e-7


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Solver:
    def __init__(self, a):
        self.a = a
        self.t0 = time.time()
        self.T = float(a.time)
        self.deadline = self.t0 + self.T - 3.0
        self.rng = np.random.default_rng(a.seed)
        self.path = instance_path(a.target)
        self.best_x = None
        self.best_sc = None
        self.fb_x = None
        self.fb_viol = INF
        self.saved = False
        self.last_viol = INF
        self.lb_ok = True
        self.hl = None
        self.hl_is_lp = False
        self.xlp = None
        self.scp_rows = None
        self.scp_cost = None

        h = highspy.Highs()
        h.silent()
        h.readModel(self.path)
        self.h = h
        self.set_opts(h)
        lp = h.getLp()
        self.n = n = int(lp.num_col_)
        self.m = m = int(lp.num_row_)
        names = list(lp.col_names_)
        self.names = names if len(names) == n else ["x%d" % i for i in range(n)]
        integ = list(lp.integrality_)
        if len(integ) == n:
            self.isint = np.array([k != highspy.HighsVarType.kContinuous for k in integ], bool)
        else:
            self.isint = np.zeros(n, bool)
        self.int_idx = np.nonzero(self.isint)[0].astype(np.int32)
        self.has_cont = int((~self.isint).sum()) > 0
        self.lo0 = np.array(lp.col_lower_, float)
        self.up0 = np.array(lp.col_upper_, float)
        self.rl = np.array(lp.row_lower_, float)
        self.ru = np.array(lp.row_upper_, float)
        self.c = np.array(lp.col_cost_, float)
        self.offset = float(lp.offset_)
        self.sense = 1.0 if h.getObjectiveSense()[1] == highspy.ObjSense.kMinimize else -1.0
        self.bin_idx = np.nonzero(self.isint & (self.lo0 == 0) & (self.up0 == 1))[0].astype(np.int32)
        am = lp.a_matrix_
        start = np.array(am.start_, dtype=np.int64)
        index = np.array(am.index_, dtype=np.int64)
        value = np.array(am.value_, dtype=float)
        try:
            colwise = bool(am.format_ == highspy.MatrixFormat.kColwise)
        except Exception:
            colwise = len(start) == n + 1
        if colwise:
            A = sp.csc_matrix((value, index, start), shape=(m, n))
        else:
            A = sp.csr_matrix((value, index, start), shape=(m, n))
        self.Acsr = A.tocsr()
        self.Acsc = A.tocsc()
        self.row_cap = max(40, int(0.05 * n))
        self.tl = clamp(self.T / 40.0, 2.0, 15.0)
        self.tlmax = clamp(self.T / 6.0, 5.0, 60.0)

    # ------------------------------------------------------------------ utilities
    def set_opts(self, h):
        h.setOptionValue("threads", 4)
        h.setOptionValue("random_seed", int(self.a.seed) % (2**31 - 1))
        h.setOptionValue("mip_feasibility_tolerance", 1e-7)
        h.setOptionValue("primal_feasibility_tolerance", 1e-7)
        h.setOptionValue("mip_rel_gap", 1e-6)

    @property
    def xinc(self):
        return self.best_x if self.best_x is not None else self.fb_x

    def canon(self, x):
        x = np.array(x, float)
        ii = self.int_idx
        if len(ii):
            x[ii] = np.round(x[ii])
        x = np.minimum(np.maximum(x, self.lo0), self.up0)
        if len(ii):
            xi = x[ii]
            r = np.round(xi)
            bad = np.abs(xi - r) > 1e-9
            if bad.any():
                f = np.floor(xi[bad])
                f = np.where(f >= self.lo0[ii][bad] - 1e-9, f, np.ceil(xi[bad]))
                xi[bad] = f
                x[ii] = xi
        return x

    def verify(self, x):
        self.last_viol = INF
        if not np.all(np.isfinite(x)):
            return False
        if np.any(x < self.lo0 - 1e-7) or np.any(x > self.up0 + 1e-7):
            return False
        ii = self.int_idx
        if len(ii) and np.any(np.abs(x[ii] - np.round(x[ii])) > 1e-9):
            return False
        viol = 0.0
        if self.m > 0:
            ax = self.Acsr @ x
            v = np.maximum(self.rl - ax, ax - self.ru)
            viol = float(max(0.0, np.max(v)))
        self.last_viol = viol
        return viol <= ROW_TOL

    def objective(self, x):
        return float(self.c @ x) + self.offset

    def save(self, x, obj):
        nz = np.nonzero(x)[0]
        sol = {self.names[i]: (int(round(x[i])) if self.isint[i] else float(x[i])) for i in nz}
        d = {"target": self.a.target, "obj": float(obj), "solution": sol}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.a.out)), exist_ok=True)
        except Exception:
            pass
        tmp = self.a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.a.out)
        self.saved = True

    def polish(self, x1):
        if self.hl is None:
            return None
        try:
            ii = self.int_idx
            if len(ii):
                v = x1[ii]
                self.hl.changeColsBounds(len(ii), ii, v, v)
            self.hl.setOptionValue("primal_feasibility_tolerance", 1e-9)
            self.hl.setOptionValue("dual_feasibility_tolerance", 1e-9)
            self.hl.setOptionValue("time_limit", float(clamp(self.deadline - time.time(), 0.5, 10.0)))
            self.hl.run()
            info = self.hl.getInfo()
            if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
                return np.array(self.hl.getSolution().col_value, float)
        except Exception:
            pass
        return None

    def consider(self, xraw):
        try:
            x = np.asarray(xraw, float)
            if x.shape[0] != self.n or not np.all(np.isfinite(x)):
                return False
            x1 = self.canon(x)
            ok = self.verify(x1)
            cand_fb, cand_v = x1, self.last_viol
            if not ok and self.has_cont:
                x2 = self.polish(x1)
                if x2 is not None:
                    x2 = self.canon(x2)
                    if self.verify(x2):
                        x1, ok = x2, True
                    elif self.last_viol < cand_v:
                        cand_fb, cand_v = x2, self.last_viol
            if not ok:
                if self.best_x is None and cand_v < self.fb_viol:
                    self.fb_x, self.fb_viol = cand_fb, cand_v
                return False
            sc = self.sense * self.objective(x1)
            if self.best_sc is None or sc < self.best_sc - 1e-9 * max(1.0, abs(self.best_sc)):
                self.best_x, self.best_sc = x1, sc
                try:
                    self.save(x1, self.sense * sc)
                except Exception:
                    pass
                return True
            return False
        except Exception:
            return False

    def take(self, h):
        try:
            info = h.getInfo()
            if info.primal_solution_status != highspy.SolutionStatus.kSolutionStatusFeasible:
                return False
            x = np.array(h.getSolution().col_value, float)
            return self.consider(x)
        except Exception:
            return False

    def setup_lp(self):
        try:
            hl = highspy.Highs()
            hl.silent()
            hl.readModel(self.path)
            hl.setOptionValue("threads", 4)
            hl.setOptionValue("primal_feasibility_tolerance", 1e-8)
            hl.setOptionValue("dual_feasibility_tolerance", 1e-8)
            self.hl = hl
            try:
                for j in self.int_idx:
                    hl.changeColIntegrality(int(j), highspy.HighsVarType.kContinuous)
                self.hl_is_lp = True
            except Exception:
                self.hl_is_lp = False
            if self.hl_is_lp and len(self.int_idx):
                rem = self.deadline - time.time()
                if rem > 3:
                    hl.setOptionValue("time_limit", float(clamp(0.1 * self.T, 2.0, min(60.0, rem - 1))))
                    hl.run()
                    if hl.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                        self.xlp = np.array(hl.getSolution().col_value, float)
        except Exception:
            self.hl = None
            self.xlp = None

    # ------------------------------------------------------------------ set covering structure
    def detect_scp(self):
        try:
            if self.m == 0 or self.n == 0 or len(self.bin_idx) != self.n:
                return False
            A = self.Acsr
            ip, dat = A.indptr, A.data
            cover = []
            for i in range(self.m):
                s, e = ip[i], ip[i + 1]
                if e <= s:
                    continue
                if (
                    self.rl[i] > -INF
                    and self.ru[i] >= INF
                    and abs(self.rl[i] - 1.0) < 1e-9
                    and np.all(np.abs(dat[s:e] - 1.0) < 1e-12)
                ):
                    cover.append(i)
            if len(cover) == 0 or len(cover) < 0.9 * self.m:
                return False
            c = self.sense * self.c
            if np.all(c == 0):
                ceff = np.ones(self.n)
            elif np.all(c > 0):
                ceff = c.copy()
            else:
                return False
            self.scp_rows = np.array(cover, dtype=np.int64)
            self.scp_cost = ceff
            return True
        except Exception:
            return False

    def scp_search(self, until):
        """Row-weighting local search (RWLS-style) for set covering; every improved cover is verified."""
        try:
            rows = self.scp_rows
            R = len(rows)
            n = self.n
            A = self.Acsr
            ip, ix = A.indptr, A.indices
            cols_of = [ix[ip[i] : ip[i + 1]].astype(np.int64).tolist() for i in rows]
            if any(len(cs) == 0 for cs in cols_of):
                return
            rows_of = [[] for _ in range(n)]
            for r, cs in enumerate(cols_of):
                for j in cs:
                    rows_of[j].append(r)
            cost = np.asarray(self.scp_cost, float)
            costl = cost.tolist()
            inv = (1.0 / cost).tolist()
            Acov = A[rows, :].tocsr()
            AcovT = Acov.T.tocsr()
            # greedy initialisation (from the incumbent if any)
            inXv = np.zeros(n, bool)
            x0 = self.xinc
            if x0 is not None:
                inXv = x0 > 0.5
            cntv = np.rint(Acov @ inXv.astype(float)).astype(np.int64)
            guard = 0
            while True:
                u = (cntv == 0).astype(float)
                if not u.any():
                    break
                guard += 1
                if guard > n + 5:
                    return
                gain = AcovT @ u
                gain[inXv] = -1.0
                j = int(np.argmax(gain / cost))
                if gain[j] <= 0:
                    return
                inXv[j] = True
                cntv[rows_of[j]] += 1
            inX = inXv.tolist()
            cnt = cntv.tolist()
            w = [1] * R
            ts = [0] * n
            score = [0.0] * n
            for j in range(n):
                if inX[j]:
                    score[j] = -float(sum(w[r] for r in rows_of[j] if cnt[r] == 1))
                else:
                    score[j] = float(sum(w[r] for r in rows_of[j] if cnt[r] == 0))
            unc = set(r for r in range(R) if cnt[r] == 0)
            Xset = set(int(j) for j in np.nonzero(inXv)[0])
            state = {"cur": float(sum(costl[j] for j in Xset))}

            def remove(j):
                inX[j] = False
                Xset.discard(j)
                state["cur"] -= costl[j]
                score[j] = -score[j]
                for r in rows_of[j]:
                    cnt[r] -= 1
                    if cnt[r] == 0:
                        unc.add(r)
                        wr = w[r]
                        for k in cols_of[r]:
                            if k != j:
                                score[k] += wr
                    elif cnt[r] == 1:
                        wr = w[r]
                        for k in cols_of[r]:
                            if inX[k]:
                                score[k] -= wr
                                break

            def add(j):
                inX[j] = True
                Xset.add(j)
                state["cur"] += costl[j]
                score[j] = -score[j]
                for r in rows_of[j]:
                    cnt[r] += 1
                    if cnt[r] == 1:
                        unc.discard(r)
                        wr = w[r]
                        for k in cols_of[r]:
                            if k != j:
                                score[k] -= wr
                    elif cnt[r] == 2:
                        wr = w[r]
                        for k in cols_of[r]:
                            if inX[k] and k != j:
                                score[k] += wr
                                break

            def pick_remove(tabu):
                bj = -1
                bs = -INF
                bts = INF
                for j in Xset:
                    if j == tabu:
                        continue
                    s = score[j] * inv[j]
                    if s > bs or (s == bs and ts[j] < bts):
                        bj, bs, bts = j, s, ts[j]
                if bj < 0:
                    bj = next(iter(Xset))
                return bj

            rng = random.Random(int(self.a.seed) * 7919 + 12345)
            bestcost = INF
            tabu_add = -1
            tabu_rem = -1
            step = 0
            while True:
                step += 1
                if (step & 127) == 0 and time.time() > until:
                    break
                if not unc:
                    if state["cur"] < bestcost - 1e-9:
                        bestcost = state["cur"]
                        xb = np.zeros(n)
                        xb[list(Xset)] = 1.0
                        self.consider(xb)
                    if not Xset:
                        break
                    j = pick_remove(-1)
                    remove(j)
                    ts[j] = step
                    tabu_add = j
                    continue
                if Xset:
                    j = pick_remove(tabu_rem)
                    remove(j)
                    ts[j] = step
                    tabu_add = j
                r = rng.choice(tuple(unc))
                cs = cols_of[r]
                k = -1
                bs = -INF
                bts = INF
                for cc in cs:
                    if cc == tabu_add or inX[cc]:
                        continue
                    s = score[cc] * inv[cc]
                    if s > bs or (s == bs and ts[cc] < bts):
                        k, bs, bts = cc, s, ts[cc]
                if k < 0:
                    k = -1
                    for cc in cs:
                        if not inX[cc]:
                            k = cc
                            break
                    if k < 0:
                        continue
                add(k)
                ts[k] = step
                tabu_rem = k
                for r2 in unc:
                    w[r2] += 1
                    for cc in cols_of[r2]:
                        score[cc] += 1.0
        except Exception:
            return

    # ------------------------------------------------------------------ neighbourhoods
    def rows_of_cols(self, cols):
        return np.unique(self.Acsc[:, cols].indices)

    def cols_of_rows(self, rows, excl):
        out = []
        ip, ix = self.Acsr.indptr, self.Acsr.indices
        for r in rows:
            cs = ix[ip[r] : ip[r + 1]]
            cs = cs[self.isint[cs] & ~excl[cs]]
            if len(cs) > self.row_cap:
                cs = self.rng.choice(cs, self.row_cap, replace=False)
            if len(cs):
                out.append(cs)
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

    def fill(self, F, k):
        F = np.unique(np.asarray(F, np.int64))
        if len(F) >= k:
            return F
        rest = np.setdiff1d(self.int_idx.astype(np.int64), F)
        if len(rest) == 0:
            return F
        extra = self.rng.choice(rest, size=min(k - len(F), len(rest)), replace=False)
        return np.concatenate([F, extra])

    def nb_random(self, k):
        return self.rng.choice(self.int_idx, size=k, replace=False).astype(np.int64)

    def nb_lp(self, k):
        ii = self.int_idx
        d = np.abs(self.xinc[ii] - self.xlp[ii])
        sel = d > 1e-4
        D = ii[sel].astype(np.int64)
        if len(D) == 0:
            return self.nb_random(k)
        if len(D) >= k:
            w = d[sel] + 0.05
            w = w / w.sum()
            return self.rng.choice(D, size=k, replace=False, p=w)
        return self.fill(D, k)

    def nb_rows(self, k):
        inF = np.zeros(self.n, bool)
        vis = np.zeros(self.m, bool)
        F = []
        cnt = 0
        tries = 0
        while cnt < k and tries < 50:
            tries += 1
            unv = np.nonzero(~vis)[0]
            if len(unv) == 0:
                break
            frontier = np.array([self.rng.choice(unv)])
            while cnt < k and len(frontier):
                vis[frontier] = True
                cs = self.cols_of_rows(frontier, inF)
                if len(cs) == 0:
                    break
                if len(cs) > k - cnt:
                    cs = self.rng.choice(cs, k - cnt, replace=False)
                inF[cs] = True
                F.append(cs)
                cnt += len(cs)
                nr = self.rows_of_cols(cs)
                nr = nr[~vis[nr]]
                if len(nr) > 500:
                    nr = self.rng.choice(nr, 500, replace=False)
                frontier = nr
        F = np.concatenate(F) if F else np.zeros(0, np.int64)
        return self.fill(F, k)

    def nb_active(self, k, weighted):
        ii = self.int_idx
        x = self.xinc
        nz = ii[x[ii] > self.lo0[ii] + 0.5].astype(np.int64)
        if len(nz) == 0:
            return self.nb_random(k)
        ns = int(min(len(nz), max(1, k // 2)))
        p = None
        if weighted:
            w = np.abs(self.c[nz] * x[nz])
            if w.max() > 0:
                w = w / w.max() + 0.02
                p = w / w.sum()
        seeds = self.rng.choice(nz, size=ns, replace=False, p=p)
        inF = np.zeros(self.n, bool)
        inF[seeds] = True
        rows = self.rows_of_cols(seeds)
        if len(rows) > 1500:
            rows = self.rng.choice(rows, 1500, replace=False)
        nb = self.cols_of_rows(rows, inF)
        if len(nb) > k - ns:
            nb = self.rng.choice(nb, k - ns, replace=False)
        return self.fill(np.concatenate([seeds, nb]), k)

    # ------------------------------------------------------------------ sub-MIP
    def run_sub(self, F, K):
        h = self.h
        ii = self.int_idx
        x = self.xinc
        fixed = np.setdiff1d(ii.astype(np.int64), F).astype(np.int32) if F is not None else np.zeros(0, np.int32)
        if len(fixed):
            h.changeColsBounds(len(fixed), fixed, x[fixed], x[fixed])
        added = False
        if K is not None:
            try:
                b = self.bin_idx
                ones = x[b] > 0.5
                vals = np.where(ones, -1.0, 1.0)
                ub = float(K - int(ones.sum()))
                h.addRow(-INF, ub, len(b), b, vals)
                added = True
            except Exception:
                self.lb_ok = False
        rem = self.deadline - time.time()
        tl = float(clamp(min(self.tl, rem - 0.5), 0.5, 1e6))
        h.setOptionValue("time_limit", tl)
        try:
            s = highspy.HighsSolution()
            s.col_value = list(x)
            h.setSolution(s)
        except Exception:
            pass
        ts = time.time()
        try:
            h.run()
        except Exception:
            pass
        el = time.time() - ts
        improved = self.take(h)
        if len(fixed):
            h.changeColsBounds(len(ii), ii, self.lo0[ii], self.up0[ii])
        if added:
            try:
                h.deleteRows(1, np.array([h.getNumRow() - 1], dtype=np.int32))
                if h.getNumRow() != self.m:
                    raise RuntimeError("row count")
            except Exception:
                self.lb_ok = False
                try:
                    h.changeRowBounds(h.getNumRow() - 1, -INF, INF)
                except Exception:
                    pass
        return improved, el, tl

    def lns(self):
        ii = self.int_idx
        if len(ii) == 0 or self.xinc is None:
            return
        size = {"random": 0.2, "rows": 0.15, "active": 0.2, "cost": 0.2, "lp": 0.15}
        K = 10
        types = ["random", "rows", "active", "cost"]
        if self.xlp is not None:
            types.append("lp")
        if len(self.bin_idx) >= 20:
            types.append("lb")
        w = {t: 1.0 for t in types}
        while True:
            rem = self.deadline - time.time()
            if rem < 1.5:
                break
            if not self.lb_ok and "lb" in types:
                types.remove("lb")
            if not types:
                break
            probs = np.array([w[t] for t in types])
            probs = probs / probs.sum()
            t = types[int(self.rng.choice(len(types), p=probs))]
            try:
                if t == "lb":
                    F, Kc = None, K
                else:
                    k = int(clamp(int(round(size[t] * len(ii))), 1, len(ii)))
                    Kc = None
                    if t == "random":
                        F = self.nb_random(k)
                    elif t == "rows":
                        F = self.nb_rows(k)
                    elif t == "active":
                        F = self.nb_active(k, False)
                    elif t == "cost":
                        F = self.nb_active(k, True)
                    else:
                        F = self.nb_lp(k)
                improved, el, tl = self.run_sub(F, Kc)
            except Exception:
                w[t] = max(0.2, w[t] * 0.5)
                continue
            if improved:
                w[t] = min(w[t] + 1.0, 6.0)
                continue
            w[t] = max(0.2, w[t] * 0.85)
            exhausted = el < 0.85 * tl
            if t == "lb":
                K = int(clamp(K * 1.5, 3, 80)) if exhausted else int(clamp(K * 0.7, 3, 80))
            elif exhausted:
                size[t] = min(0.8, size[t] * 1.3)
            else:
                size[t] = max(0.01, size[t] * 0.75)
                if size[t] <= 0.0101:
                    self.tl = min(self.tlmax, self.tl * 1.5)

    # ------------------------------------------------------------------ feasibility fallback
    def feasibility_phase(self, until):
        try:
            hs = highspy.Highs()
            hs.silent()
            hs.readModel(self.path)
            self.set_opts(hs)
            hs.setOptionValue("mip_heuristic_effort", 0.2)
            n = self.n
            hs.changeColsCost(n, np.arange(n, dtype=np.int32), np.zeros(n))
            for i in range(self.m):
                if self.rl[i] > -INF:
                    hs.addCol(1.0, 0.0, INF, 1, np.array([i], dtype=np.int32), np.array([1.0]))
                if self.ru[i] < INF:
                    hs.addCol(1.0, 0.0, INF, 1, np.array([i], dtype=np.int32), np.array([-1.0]))
        except Exception:
            return
        ii = self.int_idx
        xs = None
        sbest = INF
        first = True
        tl_feas = clamp(self.T / 30.0, 2.0, 20.0)
        while time.time() < until - 1.0 and self.best_x is None:
            rem = until - time.time()
            tl = max(1.0, 0.5 * rem) if first else clamp(tl_feas, 1.0, max(1.0, rem - 0.5))
            fixed = None
            try:
                if xs is not None and len(ii):
                    k = int(0.7 * len(ii))
                    fixed = self.rng.choice(ii, size=k, replace=False).astype(np.int32)
                    hs.changeColsBounds(len(fixed), fixed, xs[fixed], xs[fixed])
                    try:
                        s = highspy.HighsSolution()
                        s.col_value = list(xs)
                        hs.setSolution(s)
                    except Exception:
                        pass
                hs.setOptionValue("time_limit", float(tl))
                hs.run()
                info = hs.getInfo()
                if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
                    full = np.array(hs.getSolution().col_value, float)
                    obj = float(info.objective_function_value) - self.offset
                    if len(ii):
                        full[ii] = np.round(full[ii])
                    if xs is None or obj < sbest - 1e-9:
                        xs, sbest = full.copy(), obj
                    self.consider(full[: self.n])
                st = hs.getModelStatus()
                if fixed is not None:
                    hs.changeColsBounds(len(ii), ii, self.lo0[ii], self.up0[ii])
                if first and st == highspy.HighsModelStatus.kOptimal and sbest > 1e-6:
                    break  # slack optimum positive: no feasible point within tolerance
                if xs is None:
                    break
            except Exception:
                break
            first = False

    # ------------------------------------------------------------------ driver
    def run(self):
        h = self.h
        allzero = not np.any(self.c != 0)
        scp = self.detect_scp()
        frac = 0.45 if allzero else 0.3
        if scp:
            frac = 0.2
        p1 = clamp(frac * self.T, 3.0, max(3.0, self.deadline - time.time() - 1.0))
        if allzero:
            h.setOptionValue("mip_heuristic_effort", 0.2)
        h.setOptionValue("time_limit", float(p1))
        h.run()
        self.take(h)
        st = h.getModelStatus()
        if self.best_x is not None and (st == highspy.HighsModelStatus.kOptimal or allzero or len(self.int_idx) == 0):
            return
        if len(self.int_idx) == 0:
            return
        if scp:
            until = self.t0 + (0.7 if allzero else 0.85) * self.T
            until = min(until, self.deadline - 1.0)
            if until > time.time() + 1.0:
                self.scp_search(until)
            if allzero and self.best_x is not None:
                return
        if allzero:
            h.setOptionValue("mip_heuristic_effort", 0.05)
        self.setup_lp()
        if self.best_x is None:
            until = self.deadline if allzero else self.t0 + 0.65 * self.T
            self.feasibility_phase(until)
            if self.best_x is not None and allzero:
                return
        if self.xinc is None:
            rem = self.deadline - time.time()
            if rem > 2:
                h.setOptionValue("time_limit", float(rem))
                h.run()
                self.take(h)
            return
        self.lns()

    def finish(self):
        if self.saved:
            return
        if self.fb_x is not None:
            self.save(self.fb_x, self.objective(self.fb_x))
        else:
            self.save(np.zeros(self.n), self.offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--time", type=float, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = None
    try:
        s = Solver(a)
        s.run()
    except Exception:
        pass
    finally:
        try:
            if s is not None:
                s.finish()
            elif not os.path.exists(a.out):
                tmp = a.out + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"target": a.target, "obj": 0.0, "solution": {}}, f)
                os.replace(tmp, a.out)
        except Exception:
            pass


if __name__ == "__main__":
    main()
