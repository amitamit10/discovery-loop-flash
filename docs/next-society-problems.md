# Next society-valuable problem modules

Three candidates for the module after `cvrp`, each scored on the three things that make a discovery-loop
problem work: **(a)** a stable, publicly downloadable instance set; **(b)** a maintained best-known table at a
citable URL (so `records.py` can fetch the value to beat live, the way `miplib` fetches its `.solu` and `cvrp`
fetches the CVRPLIB table); **(c)** an exact, independently implementable verifier. Ranked best-fit first.

---

## 1. Unit commitment (PGLib-UC)  — recommended next

**Why it matters.** Unit commitment decides which power plants run each hour of the next day and at what
output, to meet demand at minimum fuel and start-up cost while respecting ramp rates, minimum up/down times
and reserves. Every grid operator solves it daily; a percent saved is millions of dollars and megatons of CO₂.
It is the discrete cousin of the AC-OPF already in `pglib_opf`.

- **(a) Instances:** `power-grid-lib/pglib-uc` (GitHub, actively maintained by the IEEE PES Power Grid Lib task
  force — same group as pglib-opf). Realistic day-ahead UC instances (FERC, RTS-GMLC, CAISO-derived) as JSON.
- **(b) Best-known table:** the repo README reports a reference objective per instance and the MILP is designed
  to be solved to proven optimality by a commercial MIP solver, so the "record" is an optimum, not a moving
  target. `records.py` parses the README table the way `pglib_opf/records.py` parses `BASELINE.md`. Citable,
  versioned by git tag.
- **(c) Verifier:** UC is a MILP with a published closed-form objective. The verifier re-reads the JSON, checks
  every constraint (demand balance, min up/down, ramp, reserve, generation limits) at a tolerance and recomputes
  the cost from the commitment + dispatch — pure numpy, no solver needed, exactly like `miplib/verify.py` but
  against the documented UC formulation instead of an `.mps`. HiGHS (`highspy`, already a dependency) can be the
  seed solver.
- **Credit / submission:** GitHub push is the publication (like `pglib_opf`); genuinely better bounds go to the
  pglib-uc issue tracker by hand. No email seam needed.
- **Difficulty:** **Low–medium.** The hardest part (a maintained instance + reference set with a clean exact
  verifier) is already solved by the repo. Closest in shape to modules that already exist. ~1 focused session.

---

## 2. Water distribution network design (Hanoi / New York Tunnels / Balerma, via EPANET)

**Why it matters.** Least-cost pipe sizing decides the diameter of every pipe in a water network so that every
household still gets adequate pressure. Pipes are the dominant capital cost of a water utility and last 50+
years, so a better design is direct public money and safer supply — a textbook society problem.

- **(a) Instances:** the classic least-cost design benchmarks — Hanoi (Fujiwara & Khang 1990), New York Tunnels,
  Balerma Irrigation Network (Reca & Martínez 2006), plus GoYang/Fossolo/Pescara/Modena. Widely mirrored as
  EPANET `.inp` files (e.g. bundled with `wntr`, and in several GitHub benchmark repos). Stable and small.
- **(b) Best-known table:** **this is the weak leg.** There is no single continuously-maintained authoritative
  best-known table; the widely-cited optima/best-knowns (Hanoi ≈ \$6.081M, NYT ≈ \$38.64M, Balerma ≈ \$1.923M)
  live in survey papers (Mala-Jetmarova et al. 2018; Wang et al. 2015). A module here must **pin** the best-known
  values to one cited survey and record that provenance in `BASELINE.md`, rather than fetch a live table. Fetch
  live only the instance files.
- **(c) Verifier:** exact and clean via the **`wntr`** Python package (USEPA/WNTR, wraps the EPANET 2.2 hydraulic
  engine, verified live). Given a candidate diameter per pipe, run a steady-state simulation and check pressure ≥
  the minimum at every demand node under the design demand; cost = Σ(pipe length × unit cost of chosen diameter)
  from the instance's discrete diameter/cost catalogue. The physics (mass balance + Hazen-Williams head loss) is
  solved by EPANET, so the checker is a thin, trustworthy wrapper.
- **Credit / submission:** no maintainer inbox; GitHub push is the publication, results framed against the cited
  survey's best-knowns.
- **Difficulty:** **Medium.** Verifier is easy (wntr does the hydraulics). The real work is curating a defensible
  best-known table with provenance, since (b) is not a live resource. ~1–2 sessions.

---

## 3. Nurse / staff rostering (schedulingbenchmarks.org; INRC-II alternative)

**Why it matters.** Rostering assigns shifts to staff subject to legal rest rules, skill coverage, contracts and
fairness. Hospitals run it every few weeks; a better roster is safer coverage, less burnout and less agency
overtime — a direct human-welfare problem.

- **(a) Instances:** **schedulingbenchmarks.org** (Curtois & Qu, University of Nottingham; verified live) hosts a
  large, stable, XML-format employee-scheduling benchmark that is the de-facto standard, plus the well-known
  INRC-II (Second International Nurse Rostering Competition, 2015) instance set.
- **(b) Best-known table:** schedulingbenchmarks.org **publishes and updates a best-known objective per instance**
  — the strongest (b) of the three rostering options, so it is the primary. INRC-II's best-knowns come from the
  competition ranking + follow-up papers (less actively maintained), so it is the fallback rather than the base.
- **(c) Verifier:** the objective is an explicit weighted sum of soft-constraint violations over a hard-constraint
  feasible roster; it can be recomputed exactly from the XML in pure python. INRC-II additionally ships an
  **official Java validator** that can be shelled out to as a second, authoritative cross-check (the module would
  keep a numpy/python verifier as primary and diff against the Java jar in tests).
- **Credit / submission:** GitHub push; a new best-known is reported to the benchmark maintainer by hand.
- **Difficulty:** **Medium–high.** The constraint catalogue (contracts, patterns, coverage windows, weekend
  fairness) is the most intricate of the three to implement and test to zero tolerance; the payoff is the
  best-maintained best-known table. ~2 sessions.

---

### Summary

| Candidate | (a) instances | (b) live best-known table | (c) exact verifier | difficulty |
| --- | --- | --- | --- | --- |
| Unit commitment (PGLib-UC) | strong (GitHub) | strong (repo README, optima) | numpy MILP recheck | low–medium |
| Water network design | strong (EPANET `.inp`) | weak — pin from a survey | wntr / EPANET wrapper | medium |
| Nurse rostering | strong (schedulingbenchmarks.org) | strong (published best-knowns) | python + INRC-II Java jar | medium–high |

Recommendation: **PGLib-UC next** — it reuses the `pglib_opf` shape, the record is a provable optimum, and the
verifier is a numpy constraint recheck with no new heavy dependency.
