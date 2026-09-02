# discovery-loop

An LLM evolves a solver program; a zero-tolerance checker scores it; winners survive. AlphaEvolve shape, one file.

Problems are plug-ins under `problems/<name>/problem.py` (targets, live records, independent verifier, submission format, prompt):

- **circle_packing**: Packomania csqv, pack N variable-radius circles in the unit square maximising the sum of radii.
  Records fetched live from packomania.com; zero-tolerance stdlib checker; `.pck` submissions emailed to the maintainer.
- **miplib**: MIPLIB 2017 *open* instances (real-world mixed-integer programs with no proven optimum). Best-known values from the
  official `.solu` file; independent checker re-evaluates bounds, integrality and every row at 1e-6; `.sol` files emailed to
  miplibsolutions@zib.de. Seed solver = HiGHS + adaptive LNS (`pip install highspy`).

```powershell
python loop.py --problem miplib --eval-only               # score the champion solver, no model calls
python loop.py --problem miplib --iters 20 --budget 30    # evolve (claude -p, Fable 5.1) until 20 iterations or $30
start runs-miplib\status.html                             # human view: standings, iterations, ideas tried
start runs\status.html                                    # same for circle_packing (legacy flat layout)
```

Outputs per problem: `best*/solver.py` (champion), `best*/<sub>/` (candidates in the benchmark's submission format),
`runs*/log.jsonl`, `runs*/status.html`. Verify any candidate with the problem's own `verify.py`.

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
