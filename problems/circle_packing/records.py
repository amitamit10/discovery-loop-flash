"""Fetch the live Packomania csqv record table -> records.json {N: sum_of_radii}."""

import json
import os
import urllib.request

URL = "https://www.packomania.com/csqv/txt/sumradii.txt"
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "records.json")


def fetch(path=PATH):
    txt = urllib.request.urlopen(URL, timeout=30).read().decode()
    rec = {}
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit():
            rec[int(p[0])] = float(p[1])
    json.dump(rec, open(path, "w"), indent=0)
    return rec


def load(path=PATH):
    if not os.path.exists(path):
        return fetch(path)
    return {int(k): v for k, v in json.load(open(path)).items()}


if __name__ == "__main__":
    print(f"fetched {len(fetch())} records")
