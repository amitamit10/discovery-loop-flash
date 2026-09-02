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
