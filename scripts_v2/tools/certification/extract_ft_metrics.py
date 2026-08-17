import json, sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out = {"order": ["A","B","C"], "runs": {}}
names = {"A":"2026-07-08_01-32-22_ftA","B":"2026-07-08_01-32-22_ftB","C":"2026-07-08_01-32-22_ftC"}
base = "/home/dom_iva/UWLab_jameel_port/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/"
for run, d in names.items():
    ea = EventAccumulator(base+d); ea.Reload()
    tags = ea.Tags()["scalars"]
    def series(tag):
        if tag not in tags: return [], []
        ev = ea.Scalars(tag)
        return [e.step for e in ev], [e.value for e in ev]
    x, tot = series("Metrics/task_command/end_of_episode_success_rate")
    tasks = []
    for i in range(4):
        _, v = series(f"Metrics/task_{i}_success_rate")
        tasks.append(v)
    _, prog = series("Curriculum/adr_sysid/scale_progress")
    fin = round(sum(tot[-10:])/max(1,len(tot[-10:])), 4) if tot else None
    mx = round(max(tot),4) if tot else None
    out["runs"][run] = {"x": x, "total": tot, "tasks": tasks, "adr": prog,
        "final": fin, "max": mx,
        "display_name": f"Stage {run} finetune (sysid 0.575kg)", "run_name": d,
        "label": f"ft{run}", "sub": f"final {fin} @ full DR"}
    sys.stderr.write(f"{run}: n={len(x)} final={fin} max={mx}\n")
json.dump(out, open("/home/dom_iva/ft_metrics.json","w"))
