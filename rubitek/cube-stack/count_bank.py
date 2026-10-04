"""Count states in a reset bank. No Isaac, no GPU.

A byte-size threshold cannot tell a COMPLETE bank from a TRUNCATED one -- a killed recording left a
207 KB file that passed a >=100 KB check while holding ~49 of a requested 2000 states. Counting the
first dimension of the stored tensors is the only honest check.
"""
import sys, torch
from pathlib import Path

for f in sys.argv[1:]:
    p = Path(f)
    if not p.exists():
        print(f"{p.name:46} MISSING")
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    n = None
    init = d.get("initial_state", d)
    for group in ("articulation", "rigid_object"):
        if isinstance(init, dict) and group in init:
            for _, entity in init[group].items():
                for _, v in entity.items():
                    n = len(v) if isinstance(v, list) else int(v.shape[0])
                    break
                break
            break
    if n is None and isinstance(d, dict):
        for _, v in d.items():
            if hasattr(v, "shape"):
                n = int(v.shape[0]); break
    print(f"{p.name:46} {p.stat().st_size/1e6:7.2f} MB   {n if n is not None else '?'} states")
