"""Print the reward weights/stds actually saved in a run's env.yaml.

Reads the SAVED config, not the command line: a hydra key that silently fails to apply leaves the
CLI echo looking perfect while the run trains on the old value.
"""
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
for term in ("position_tracking", "orientation_tracking", "success"):
    m = re.search(r"^  %s:\n(.*?)(?=^  \w+:\n)" % term, text, re.S | re.M)
    if not m:
        print(term, "NOT FOUND")
        continue
    block = m.group(1)
    std = re.findall(r"^\s+(?:pos_std|rot_std|std):\s*(\S+)", block, re.M)
    weight = re.findall(r"^\s+weight:\s*(\S+)", block, re.M)
    print("%-22s weight=%s std=%s" % (term, weight, std))
