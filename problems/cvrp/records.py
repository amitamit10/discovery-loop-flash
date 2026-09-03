"""CVRPLIB X best-known solutions (Uchoa et al. 2017) and instance download.

The X set has 100 instances; the loop targets ten whose best known solution is NOT proven optimal
(the "Optimal" column on the live table reads "no"), so beating the value is a real result. The table
(best-known cost, optimality flag, and the site's per-instance download ids) is fetched live from the
CVRPLIB instances page and cached in records.json; a "better" result must clear the integer best-known
by at least half a unit (see problem.beats). Instance .vrp / .sol files download on first use.

This module deliberately does NOT read BASELINE.md: unlike pglib_opf (whose records.py parses its
BASELINE.md), here BASELINE.md only documents the seed solver's gaps and the record values come from
the live table below.
"""

import json
import os
import re
import urllib.request

SITE = "https://galgos.inf.puc-rio.br/cvrplib"
TABLE_URL = SITE + "/en/instances"
HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCES = os.path.join(HERE, "instances")
RECORDS = os.path.join(HERE, "records.json")

# The ten targets: X instances with 200 <= (nodes in the name) <= 500 whose best known is not proven optimal.
TARGETS = [
    "X-n280-k17",
    "X-n303-k21",
    "X-n327-k20",
    "X-n336-k84",
    "X-n401-k29",
    "X-n411-k19",
    "X-n429-k61",
    "X-n459-k26",
    "X-n480-k70",
    "X-n491-k59",
]


def _num(s):
    """Table numbers are rendered like $100$ or $27{,}591.00$."""
    return float(s.replace("{,}", "").replace(",", ""))


def parse_table(html):
    """{name: {bks, optimal, instance_id, bks_id, customers, k, capacity}} for every X-n*-k* row."""
    rec = {}
    blocks = re.split(r'(?=<a href="/cvrplib/en/download/instance/\d+")', html)
    for b in blocks:
        mid = re.search(r'download/instance/(\d+)"[^>]*title="Instance File"', b)
        mname = re.search(r">\s*(X-n\d+-k\d+)\s*<", b)
        if not (mid and mname):
            continue
        nums = re.findall(r"\$([0-9][0-9.,{}]*)\$", b)  # customers, k, capacity, then the bks
        mbks = re.search(r'download/bks/(\d+)"[^$]*\$([0-9][0-9.,{}]*)\$', b, re.S)
        mopt = re.search(r"badge[^>]*>\s*(yes|no)\s*<", b)
        if len(nums) < 3 or not mbks or not mopt:
            continue
        rec[mname.group(1)] = {
            "customers": int(_num(nums[0])),
            "k": int(_num(nums[1])),
            "capacity": int(_num(nums[2])),
            "bks": _num(mbks.group(2)),
            "optimal": mopt.group(1) == "yes",
            "instance_id": int(mid.group(1)),
            "bks_id": int(mbks.group(1)),
        }
    return rec


def _validate(rec, where):
    if len(rec) < 90:
        raise RuntimeError(f"CVRPLIB table {where}: only {len(rec)} X instances parsed (expected ~100)")
    missing = [t for t in TARGETS if t not in rec]
    if missing:
        raise RuntimeError(f"CVRPLIB table {where}: targets missing from table: {missing}")
    return rec


def fetch_table():
    """Download the live instances page, parse it, cache the full metadata table to records.json."""
    html = urllib.request.urlopen(TABLE_URL, timeout=60).read().decode("utf-8", "replace")
    rec = _validate(parse_table(html), "live")
    json.dump(rec, open(RECORDS, "w"), indent=1)
    return rec


def table():
    """Full metadata table: cached records.json if present, else fetched live."""
    if os.path.exists(RECORDS):
        return _validate(json.load(open(RECORDS)), "cache")
    return fetch_table()


def load():
    """{name: best-known cost} for the loop (the value to beat)."""
    return {k: v["bks"] for k, v in table().items()}


def fetch():
    return {k: v["bks"] for k, v in fetch_table().items()}


def _download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    open(path, "w", encoding="utf-8", newline="\n").write(data)
    return data


def instance_path(name):
    """Local .vrp for an instance, downloading from CVRPLIB on first use. The site's download ids are
    database keys, so re-read the file's own NAME field and reject a mismatch (a stale id would otherwise
    hand back the wrong coordinates and any 'record' checked against them would be bogus)."""
    p = os.path.join(INSTANCES, name + ".vrp")
    if not os.path.exists(p):
        iid = table()[name]["instance_id"]
        text = _download(f"{SITE}/en/download/instance/{iid}", p)
        got = re.search(r"NAME\s*:\s*(\S+)", text)
        if not got or got.group(1) != name:
            os.remove(p)
            raise RuntimeError(f"download id {iid} returned instance {got and got.group(1)!r}, expected {name!r}")
    return p


def official_solution_path(name):
    """Local copy of the published best-known .sol, downloaded on first use (used by the ground-truth check)."""
    p = os.path.join(INSTANCES, name + ".sol")
    if not os.path.exists(p):
        _download(f"{SITE}/en/download/bks/{table()[name]['bks_id']}", p)
    return p


if __name__ == "__main__":
    import sys

    t = fetch_table() if "--fetch" in sys.argv else table()
    n_open = sum(not v["optimal"] for v in t.values())
    print(f"{len(t)} X instances, {n_open} not proven optimal; {len(TARGETS)} targets")
    for name in TARGETS:
        v = t[name]
        print(
            f"  {name:12} customers={v['customers']:4} k={v['k']:3} Q={v['capacity']:5} bks={v['bks']:.0f} optimal={v['optimal']}"
        )
