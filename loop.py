"""Discovery loop: an LLM evolves a solver program; a code checker scores it; winners survive.

Problem: pack N variable-radius circles in the unit square, maximise the sum of radii (Packomania csqv).
Each iteration: prompt the model with the champion solver + scoreboard + ideas tried -> new solver.py
-> run it on every target N in parallel (hard timeout) -> verify with zero tolerance -> keep if better.
Per-N record candidates are saved as best/pck/csqvN.pck regardless of which solver produced them.

Usage:
  python loop.py --eval-only                     # score the current champion, no LLM calls
  python loop.py --iters 40 --budget 30          # evolve for 40 iterations or $30 of model usage
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import html
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "problems", "circle_packing"))
import verify
import records  # noqa: E402

BEST = os.path.join(HERE, "best")
CHAMP = os.path.join(BEST, "solver.py")
SCORES = os.path.join(BEST, "scores.json")
LOG = os.path.join(HERE, "runs", "log.jsonl")
STATUS = os.path.join(HERE, "runs", "status.html")
AUTHOR = "Wes Sander, MoltFire"
DEFAULT_NS = "26,32,101,102,103,105,106,107,108,109,111,114"


def run_solver(solver, n, budget, seed, out):
    t0 = time.time()
    try:
        p = subprocess.run(
            [
                sys.executable,
                solver,
                "--n",
                str(n),
                "--time",
                str(budget),
                "--seed",
                str(seed),
                "--out",
                out,
            ],
            capture_output=True,
            text=True,
            timeout=budget + 45,
        )
        if p.returncode != 0:
            return {"n": n, "sum": 0.0, "error": (p.stderr or p.stdout)[-600:]}
        d = json.load(open(out))
        res = verify.check(d["circles"], n)
        if not res["feasible"]:
            return {"n": n, "sum": 0.0, "error": "infeasible: " + json.dumps(res)[:300]}
        return {
            "n": n,
            "sum": res["sum"],
            "circles": d["circles"],
            "secs": round(time.time() - t0, 1),
        }
    except subprocess.TimeoutExpired:
        return {"n": n, "sum": 0.0, "error": f"timeout after {budget + 45}s"}
    except Exception as e:  # bad JSON, missing file, ...
        return {"n": n, "sum": 0.0, "error": f"{type(e).__name__}: {e}"[:300]}


def evaluate(solver, ns, budget, seed, workdir, workers):
    os.makedirs(workdir, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = [
            ex.submit(
                run_solver,
                solver,
                n,
                budget,
                seed + n,
                os.path.join(workdir, f"n{n}.json"),
            )
            for n in ns
        ]
        return [f.result() for f in futs]


def update_bests(results, iteration, rec):
    """Persist per-N bests + .pck files. Returns list of Ns that improved and Ns beating the record."""
    scores = json.load(open(SCORES)) if os.path.exists(SCORES) else {}
    improved, wins = [], []
    for r in results:
        if r["sum"] <= 0:
            continue
        k = str(r["n"])
        if r["sum"] > scores.get(k, {}).get("sum", 0):
            scores[k] = {"sum": r["sum"], "iter": iteration, "record": rec.get(r["n"])}
            improved.append(r["n"])
            os.makedirs(os.path.join(BEST, "pck"), exist_ok=True)
            open(os.path.join(BEST, "pck", f"csqv{k}.pck"), "w").write(verify.to_pck(r["circles"], AUTHOR))
            json.dump(
                {"n": r["n"], "circles": r["circles"]},
                open(os.path.join(BEST, "pck", f"csqv{k}.json"), "w"),
            )
        if rec.get(r["n"]) is not None and r["sum"] > rec[r["n"]]:
            wins.append(r["n"])
    json.dump(scores, open(SCORES, "w"), indent=1)
    return improved, wins


def scoreboard(ns, rec, last=None):
    scores = json.load(open(SCORES)) if os.path.exists(SCORES) else {}
    rows = []
    for n in ns:
        b = scores.get(str(n), {}).get("sum")
        r = rec.get(n)
        l = next((x["sum"] for x in (last or []) if x["n"] == n), None)
        delta = (b - r) if (b and r) else None
        rows.append((n, r, b, l, delta))
    return rows


def build_prompt(ns, rec, last_results, history):
    champ = open(CHAMP).read()
    rows = scoreboard(ns, rec, last_results)
    board = "N | packomania record | our best | champion last run | best-record\n"
    for n, r, b, l, d in rows:
        board += f"{n} | {r if r else '-'} | {b if b else '-'} | {l if l is not None else '-'} | {('%+.6f' % d) if d is not None else '-'}\n"
    hist = (
        "\n".join(
            f"iter {h['iter']}: total={h['total']:.4f} ({h['status']}) IDEA: {h['idea']}"
            + (f" ERRORS: {h['errors']}" if h.get("errors") else "")
            for h in history[-12:]
        )
        or "(none yet)"
    )
    return f"""You are evolving a Python solver for the Packomania csqv benchmark:
pack N circles of variable radius in the unit square [0,1]^2, no two overlapping, all fully inside, MAXIMISE the sum of radii.
Best-known records are tight; wins come from better optimisation, not tricks. Every packing is checked with zero tolerance.

INTERFACE CONTRACT (keep exactly):
  python solver.py --n N --time SECONDS --seed S --out PATH
  writes JSON {{"n": N, "circles": [[x, y, r], ...]}} (corner convention, square is [0,1]^2)
  must finish within SECONDS (hard kill at SECONDS+45; returning early is fine); print nothing important to stdout
  allowed imports: python stdlib, numpy, scipy (torch with an 8GB CUDA GPU is available but optional)
  the result must be STRICTLY feasible in float64 (wall slack >= 0, d^2 >= (ri+rj)^2 for all pairs, r > 0); keep a final shrink step
  a timeout, crash, or infeasible output scores 0 for that N, so reliability beats ambition

CURRENT CHAMPION solver.py:
```python
{champ}
```

SCOREBOARD (target Ns; totals are summed over these Ns):
{board}
IDEAS TRIED SO FAR:
{hist}

TASK: write a complete replacement solver.py that raises the total sum of radii across the target Ns.
Make one substantive algorithmic improvement (or a coherent combination). Candidates: smarter initialisation (hex/square lattices with
defect circles, corner-first large circles, reuse of the N-1 structure), basin hopping / perturb-and-repolish of the incumbent instead of
cold restarts, active-set Newton or SLSQP polish on the contact graph after the penalty phase, alternating LP-radii / centre moves,
swap/relocate moves for the smallest circles into the largest holes, adaptive penalty schedules, using the full time budget on the
best basin rather than many weak restarts. Do not repeat an idea that already failed unless you fix its specific failure.

OUTPUT FORMAT: first line "IDEA: <one sentence>", then exactly one ```python block with the full file. Nothing else."""


def call_model(prompt, model):
    env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC_"))}
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--max-turns",
        "1",
        "--setting-sources",
        "",
        "--tools",
        "",
        "--no-session-persistence",
        "--system-prompt",
        "You are an expert in numerical optimisation and computational geometry. Output only what is asked.",
    ]
    p = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        shell=(os.name == "nt"),
    )
    try:
        j = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None, 0.0, f"bad cli output: {p.stdout[-300:]} {p.stderr[-300:]}"
    text, cost = j.get("result") or "", float(j.get("total_cost_usd") or 0)
    m = re.search(r"```python\s*\n(.*?)```", text, re.S)
    idea = re.search(r"IDEA:\s*(.+)", text)
    return (
        (m.group(1) if m else None),
        cost,
        (idea.group(1).strip() if idea else "(no idea line)"),
    )


def write_status(ns, rec, history, cost_total, champ_total):
    rows = scoreboard(ns, rec)
    tr = ""
    for n, r, b, l, d in rows:
        cls = "win" if (d is not None and d > 0) else ("new" if (b and r is None) else "")
        tr += (
            f"<tr class='{cls}'><td>{n}</td><td>{r if r else '(no entry)'}</td><td>{b if b else '-'}</td>"
            f"<td>{('%+.6f' % d) if d is not None else '-'}</td></tr>"
        )
    hist = "".join(
        f"<tr><td>{h['iter']}</td><td>{h['total']:.4f}</td><td>{h['status']}</td>"
        f"<td>${h['cost']:.2f}</td><td>{html.escape(h['idea'])}</td></tr>"
        for h in reversed(history)
    )
    open(STATUS, "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8><title>discovery-loop</title>
<style>body{{font:14px system-ui;margin:2em;max-width:60em}}table{{border-collapse:collapse;margin:1em 0}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
th{{background:#eee}}.win{{background:#c8f7c5}}.new{{background:#fff3b0}}h1{{margin:0}}</style>
<h1>discovery-loop: circle packing (Packomania csqv)</h1>
<p>Updated {time.strftime("%Y-%m-%d %H:%M")} · champion total {champ_total:.4f} · model spend ${cost_total:.2f} · green = beats listed record · yellow = no table entry yet</p>
<table><tr><th>N</th><th>record</th><th>ours</th><th>ours - record</th></tr>{tr}</table>
<h3>Iterations</h3><table><tr><th>#</th><th>total</th><th>status</th><th>cost</th><th style='text-align:left'>idea</th></tr>{hist}</table>
<p>Record candidates: <code>best/pck/csqvN.pck</code> (Packomania submission format). Verify any file with <code>python problems/circle_packing/verify.py best/pck/csqvN.json</code>.</p>""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default=DEFAULT_NS)
    ap.add_argument("--time", type=float, default=120, help="seconds per N per solver run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--budget", type=float, default=30.0, help="max model spend in USD")
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--refresh-records", action="store_true")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    rec = records.fetch() if a.refresh_records else records.load()
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    history = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    cost_total = sum(h["cost"] for h in history)
    champ_total = max(
        [h["total"] for h in history if h["status"] in ("champion", "seed")],
        default=0.0,
    )
    it = (history[-1]["iter"] + 1) if history else 0

    def log(entry):
        history.append(entry)
        open(LOG, "a").write(json.dumps(entry) + "\n")
        write_status(ns, rec, history, cost_total, champ_total)

    if a.eval_only or not history:
        res = evaluate(
            CHAMP,
            ns,
            a.time,
            1000 * it,
            os.path.join(HERE, "runs", f"iter{it:03d}"),
            a.workers,
        )
        total = sum(r["sum"] for r in res)
        improved, wins = update_bests(res, it, rec)
        champ_total = max(champ_total, total)
        log(
            {
                "iter": it,
                "total": total,
                "status": "seed",
                "cost": 0.0,
                "idea": "seed solver (penalty L-BFGS-B + LP radii)",
                "errors": "; ".join(f"N{r['n']}: {r['error']}" for r in res if r.get("error")),
                "wins": wins,
                "improved": improved,
            }
        )
        print(f"[iter {it}] seed total={total:.4f} wins={wins} improved={improved}")
        it += 1
        if a.eval_only:
            return
    last_results = None
    while it < a.iters and cost_total < a.budget:
        prompt = build_prompt(ns, rec, last_results, history)
        code, cost, idea = call_model(prompt, a.model)
        cost_total += cost
        if not code:
            log(
                {
                    "iter": it,
                    "total": 0.0,
                    "status": "no-code",
                    "cost": cost,
                    "idea": idea,
                    "errors": "",
                }
            )
            print(f"[iter {it}] model returned no code ({idea})")
            it += 1
            continue
        wd = os.path.join(HERE, "runs", f"iter{it:03d}")
        os.makedirs(wd, exist_ok=True)
        cand = os.path.join(wd, "solver.py")
        open(cand, "w", encoding="utf-8").write(code)
        res = evaluate(cand, ns, a.time, 1000 * it, wd, a.workers)
        total = sum(r["sum"] for r in res)
        improved, wins = update_bests(res, it, rec)
        errors = "; ".join(f"N{r['n']}: {r['error']}" for r in res if r.get("error"))
        if total > champ_total:
            champ_total = total
            status = "champion"
            open(CHAMP, "w", encoding="utf-8").write(code)
        else:
            status = "rejected"
        last_results = res
        log(
            {
                "iter": it,
                "total": total,
                "status": status,
                "cost": cost,
                "idea": idea,
                "errors": errors[:800],
                "wins": wins,
                "improved": improved,
            }
        )
        print(
            f"[iter {it}] {status} total={total:.4f} champ={champ_total:.4f} cost=${cost_total:.2f} wins={wins} improved={improved} | {idea}"
        )
        it += 1
    print(f"done. champion total={champ_total:.4f} spend=${cost_total:.2f}")


if __name__ == "__main__":
    main()
