"""Minimal MATPOWER case-file (.m) parser: the numeric matrices PGLib-OPF cases use, as numpy arrays.

    case = load("pglib_opf_case5_pjm.m")
    case["baseMVA"], case["bus"], case["gen"], case["branch"], case["gencost"], case["name"]

Column layouts follow the MATPOWER manual (bus 13 cols, gen 10+, branch 13, gencost 4+n). gencost rows are
left-aligned and right-padded with zeros so rows with different polynomial degrees share one array
(MATPOWER convention: coefficients c(n-1)..c0 sit in columns 4..4+n-1, read n from column 3).
"""

import re

import numpy as np

_MATRIX = re.compile(r"mpc\.(\w+)\s*=\s*\[(.*?)\];", re.S)
_SCALAR = re.compile(r"mpc\.(\w+)\s*=\s*([\d.eE+-]+)\s*;")
_NAME = re.compile(r"function\s+mpc\s*=\s*(\w+)")

# column indices used elsewhere (0-based)
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX = range(13)
MODEL, STARTUP, SHUTDOWN, NCOST, COST = range(5)


def _strip_comments(text):
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def _rows(body):
    rows = []
    for line in body.replace(";", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.replace(",", " ").split()])
    return rows


def _matrix(rows):
    width = max(len(r) for r in rows)
    out = np.zeros((len(rows), width))
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def load(path):
    text = _strip_comments(open(path, encoding="utf-8", errors="replace").read())
    case = {"name": (_NAME.search(text) or [None, path])[1]}
    for key, val in _SCALAR.findall(text):
        case[key] = float(val)
    for key, body in _MATRIX.findall(text):
        rows = _rows(body)
        if rows:
            case[key] = _matrix(rows)
    for req in ("baseMVA", "bus", "gen", "branch", "gencost"):
        if req not in case:
            raise ValueError(f"{path}: missing mpc.{req}")
    if case["gencost"].shape[0] != case["gen"].shape[0]:
        raise ValueError(f"{path}: {case['gen'].shape[0]} generators but {case['gencost'].shape[0]} cost rows")
    return case


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        c = load(p)
        print(
            f"{c['name']}: baseMVA={c['baseMVA']} bus={c['bus'].shape} gen={c['gen'].shape} "
            f"branch={c['branch'].shape} gencost={c['gencost'].shape}"
        )
