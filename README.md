# discovery-loop

An LLM evolves a solver program; a zero-tolerance checker scores it; winners survive. AlphaEvolve shape, one file.

First problem: **Packomania csqv**, pack N variable-radius circles in the unit square maximising the sum of radii.
Records are fetched live from packomania.com and every packing is verified with an independent stdlib checker.

```powershell
python loop.py --eval-only                # score the champion solver, no model calls
python loop.py --iters 40 --budget 30     # evolve (claude -p, Fable 5.1) until 40 iterations or $30
start runs\status.html                    # human view: per-N standings, iterations, ideas tried
```

Outputs: `best/solver.py` (champion), `best/pck/csqvN.pck` (record candidates in Packomania submission format),
`runs/log.jsonl`, `runs/status.html`. Verify any candidate: `python problems/circle_packing/verify.py best/pck/csqvN.json`.

## Publishing (nothing sits on this machine)

`publish.py` runs automatically after any iteration that improves a record-beating N, and at the end of a run:

1. commits and pushes `best/` to GitHub (public, timestamped ledger of every candidate)
2. emails the new `.pck` files to Packomania through the governed `invoke-capability send-email` seam:
   DashClaw opens a pending approval, Wes approves on Telegram, `moltfire@practicalsystems.io` sends with Wes cc'd.
   Every candidate is re-verified against the live table first; `best/submitted.json` records what went out; one email per 12h.

```powershell
python publish.py --dry-run   # show what would go out
python publish.py             # push + request approval for anything new
```
