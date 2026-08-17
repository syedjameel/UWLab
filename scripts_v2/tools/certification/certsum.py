import json, sys
for tag in ("smoke_lift", "reorient_ep1700", "tilt03_ep1800"):
    try:
        d = json.load(open("/home/dom_iva/cert_%s.json" % tag))
    except Exception as e:
        print(tag, "MISSING", e)
        continue
    s = d.get("summary", d)
    print(tag, "mean_steps=", s.get("mean_episode_steps"), "terminal=", s.get("terminal_counts"))
