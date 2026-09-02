"""Publish verified record-beating packings so they never just sit on this machine.

1. git commit + push best/ (GitHub = public, timestamped ledger of every candidate)
2. email new .pck files to Packomania through the governed invoke-capability seam:
   DashClaw pending approval -> Wes approves on Telegram -> moltfire@ sends, Wes cc'd.

loop.py runs this after any iteration that improves a record-beating N. By hand:
  python publish.py            # push + email anything new (12h cooldown between emails)
  python publish.py --dry-run  # show what would go out; no push, no approval request
  python publish.py --force    # ignore the cooldown
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "problems", "circle_packing"))
import records  # noqa: E402
import verify  # noqa: E402

PCK = os.path.join(HERE, "best", "pck")
SCORES = os.path.join(HERE, "best", "scores.json")
LEDGER = os.path.join(HERE, "best", "submitted.json")
LOCK = os.path.join(HERE, "runs", "publish.lock")
CLAWD = os.environ.get("CLAWD_ROOT", os.path.expanduser("~/clawd"))
INVOKE = os.path.join(CLAWD, "agent-comms", "team", "bin", "invoke-capability.mjs")
TO = "eckard.specht@ovgu.de"  # Packomania maintainer, mailto on packomania.com
REPO_URL = "https://github.com/ucsandman/discovery-loop"
COOLDOWN = 12 * 3600  # seconds between emails to a human maintainer
APPROVAL_WAIT = 4 * 3600  # seconds to wait for Wes's approval
MIN_GAIN = 1e-6  # re-submit an N only if it improved by at least this much


def sh(*cmd, cwd=HERE):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def candidates(rec, ledger):
    """Ns where our re-verified packing beats the live record and is new since the last submission."""
    scores = json.load(open(SCORES)) if os.path.exists(SCORES) else {}
    out = []
    for k, s in scores.items():
        n, r = int(k), rec.get(int(k))
        if r is None or s["sum"] <= r:
            continue
        res = verify.check(json.load(open(os.path.join(PCK, f"csqv{k}.json")))["circles"], n)
        if not res["feasible"]:
            print(f"skip N={n}: stored packing failed re-verification")
            continue
        if res["sum"] <= ledger.get(k, {}).get("sum", 0) + MIN_GAIN:
            continue
        out.append((n, res["sum"], r))
    return sorted(out)


def git_push(dry):
    if not sh("git", "remote").stdout.strip():
        print("push: no git remote configured")
        return
    if dry:
        print("push: (dry run)")
        return
    sh("git", "add", "best")
    if sh("git", "diff", "--cached", "--quiet").returncode != 0:
        sh("git", "commit", "-q", "-m", f"best: record candidates {time.strftime('%Y-%m-%d %H:%M')}")
    p = sh("git", "push", "-q")
    print("push:", "ok" if p.returncode == 0 else p.stderr[-300:])


def body_for(cands):
    rows = "\n".join(f"  N={n:<4d} ours {s:.12f}   listed {r:.12f}   +{s - r:.2e}" for n, s, r in cands)
    plural = "s" if len(cands) > 1 else ""
    return f"""Dear Eckard Specht,

attached are {len(cands)} candidate packing{plural} for the csqv table
(circles of variable radii in the unit square, maximising the sum of radii), in the .pck
format from your hints page: square of side 1 centred at the origin, one "x y r" line per
circle sorted by increasing radius, 16 decimals.

{rows}

The packings come from an LLM-evolved optimiser (penalty L-BFGS-B, basin hopping on the
incumbent, contact-graph SLSQP polish) and were checked with an independent zero-tolerance
verifier: every circle strictly inside the square and d^2 >= (ri+rj)^2 for every pair in
float64. Authors: Wes Sander, MoltFire (AI agent).

Code, checker and every candidate: {REPO_URL}

Thank you for maintaining Packomania.

Wes Sander
"""


def email(cands, ledger, dry, force):
    now = time.time()
    if not force and now - ledger.get("_last_email", 0) < COOLDOWN:
        print(f"email: cooldown, last attempt {(now - ledger['_last_email']) / 3600:.1f}h ago")
        return
    if os.path.exists(LOCK) and now - os.path.getmtime(LOCK) < APPROVAL_WAIT + 600:
        print("email: another publish is still waiting for approval")
        return
    ns = ",".join(str(n) for n, _, _ in cands)
    inp = json.dumps(
        {
            "to": TO,
            "subject": f"csqv: {len(cands)} new candidate packing{'s' if len(cands) > 1 else ''} (N={ns})",
            "body": body_for(cands),
            "attachments": ",".join(os.path.join(PCK, f"csqv{n}.pck") for n, _, _ in cands),
        }
    )
    cmd = ["node", INVOKE, "send-email", "--agent", "moltfire", "--input", inp, "--timeout", str(APPROVAL_WAIT)]
    if dry:
        print(body_for(cands))
        p = sh(*cmd, "--dry-run", cwd=CLAWD)
        print(p.stdout or p.stderr)
        return
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    open(LOCK, "w").write(str(os.getpid()))
    ledger["_last_email"] = now
    json.dump(ledger, open(LEDGER, "w"), indent=1)
    print(f"email: requesting approval for N={ns} (waits up to {APPROVAL_WAIT // 3600}h)", flush=True)
    try:
        p = sh(*cmd, cwd=CLAWD)
    finally:
        os.remove(LOCK)
    if p.returncode != 0:
        print("email: NOT sent:", (p.stderr or p.stdout)[-400:])
        return
    action = json.loads(p.stdout).get("action_id")
    sent_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    for n, s, r in cands:
        ledger[str(n)] = {"sum": s, "listed": r, "sent_at": sent_at, "action_id": action}
    json.dump(ledger, open(LEDGER, "w"), indent=1)
    print(f"email: sent to {TO}, action {action}, N={ns}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(f"publish {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        rec = records.fetch()
    except Exception as e:  # offline: never claim a win against a stale table without saying so
        rec = records.load()
        print(f"records: live fetch failed ({e}), using cached table")
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {}
    cands = candidates(rec, ledger)
    git_push(a.dry_run)
    if not cands:
        print("email: nothing new beats the live table")
        return
    email(cands, ledger, a.dry_run, a.force)


if __name__ == "__main__":
    main()
