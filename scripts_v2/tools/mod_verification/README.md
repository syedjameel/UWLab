# 5090-migration verification scripts (bead UWLab-qiao.1 follow-on)

These are the exact scripts that produced every measured number reported for the C1-C3, D, E1,
F, and G2 runs on the vast.ai 5090 (instance 47838274) during the 2026-08-17 migration and
mod 1 / mod 2 verification pass. Until now they lived only in an ephemeral agent `/tmp`
scratchpad -- vendored here on instruction, for the same reason `../asset_authoring/` exists:
these are the artifacts that justify every number being reported, and losing them would mean
nobody could re-run or audit the checks that backed those numbers.

Every script is self-contained: it sets its own `os.environ[...]` toggles at the top (no shell
export required beyond `OMNI_KIT_ACCEPT_EULA` / `ACCEPT_EULA`), boots `isaaclab.app.AppLauncher`,
constructs the DexLift TableLeg env directly (not through `generate_reset_states_policy.py`), and
prints plain `KEY=value` lines meant to be grepped out of a log file. None of them write to any
live/production asset path; all outputs are either stdout or (E1/C-scripts) nothing on disk at all.

## What each one checks and what it found

| Script | Toggle under test | Task | Result |
|---|---|---|---|
| `gate_a3_stepcheck.py` | none -- box behavioural gate | n/a, bare `SimulationContext` | Isaac boots and steps physics on this GPU. STEPPED_10_OK. |
| `gate_b1_shadow_check.py` | none -- PYTHONPATH shadowing proof | n/a | Confirms `uwlab_tasks`/`uwlab_assets` resolve from the transferred tree, not the box's conflicting `/root/simdist/repo` editable install, by booting Isaac and printing `__file__` + the registered DexLift task ids. |
| `c1_baseline.py` | both new toggles UNSET | TableLeg-Reorient-v0 | Control run. Reference plant confirmed (effort [30.0], velocity [10000.0]). PASS. |
| `c2_clearance.py` | `DEXLIFT_SPAWN_CLEARANCE=1` | TableLeg-Reorient-v0 | Measures achieved spawn clearance from the LIVE post-reset pose via the same origin-offset-aware formula `dexlift/mdp/spawn.py` ships. min=0.0213, zero below the 1cm floor -- the 6.2mm centroid-offset fix confirmed live in the built cfg, not just the unit test. |
| `c3_partial_assembly.py` | `DEXLIFT_PARTIAL_ASSEMBLY=1` | TableLeg-Reorient-v0 | Measures fixture placement, a compose/decompose round-trip against the live `partial_assemblies.pt` bank, and goal-vs-object error, all AFTER 5 settle steps. Found a large (11deg mean, 130deg max) orientation error -- see `e1_goal_at_t0.py`, which showed this was settle drift, not a bug. |
| `e1_goal_at_t0.py` | `DEXLIFT_PARTIAL_ASSEMBLY=1` | TableLeg-Reorient-v0 | Same goal-vs-object measurement as C3 but at t=0 (zero sim steps after reset) -- the only moment "goal == spawn pose" is actually claimed to hold. Position error exactly 0.0, orientation error ~1e-7 rad (float noise) across all 32 envs. Confirms `GoalAtSpawnPoseCommand` is correct; C3's numbers were settle drift. |

Steps D, F, and G2 (generation runs with the real checkpoint, `--num_reset_states 300`) used the
vendored `../generate_reset_states_policy.py` directly, not a script from this directory -- no
separate script needed there, only CLI args and env vars. Step F was killed mid-run when its
[verify] line showed the toggle hadn't taken effect (`DexLift...TableLeg...ReorientEnvCfg_PLAY`
does not subclass the class carrying `DEXLIFT_PARTIAL_ASSEMBLY`'s `__post_init__` -- see that
run's report for the exact file:line). G2 hit the SAME class-hierarchy gap for
`DEXLIFT_GOAL_AT_SPAWN` but was caught automatically by a hard guard added to
`generate_reset_states_policy.py` after F, and raised before writing any output.

## Running one of these

Needs a real Isaac Sim + IsaacLab + `uwlab_tasks`/`uwlab_assets` environment on `PYTHONPATH`
(these scripts do not set `PYTHONPATH` themselves -- that has to come from the caller, matching
every other launcher in this project):

```
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONPATH="<repo>/source/uwlab:<repo>/source/uwlab_tasks:<repo>/source/uwlab_assets:<repo>/source/uwlab_rl"
python -u scripts_v2/tools/mod_verification/c2_clearance.py > c2.log 2>&1
```

No CLI args -- every parameter (num_envs, task id, which toggle) is a constant near the top of
each file, because these were written as one-shot verification runs, not reusable tools. Edit the
constant if you need a different task id or env count; there is no argparse here (unlike
`generate_reset_states_policy.py`, which is a real reusable tool).

## Safety note carried over from every run of these on a shared/rented box

Isaac's `simulation_app.close()` hangs routinely -- a script printing its own `..._DONE` marker and
returning is NOT proof the process has exited. Always follow a run with a process check
(`ps aux | grep -iE 'python|kit|isaac'`) and kill any survivor by literal PID in a SEPARATE
command, never with a `-f` pattern in the same command as the kill.
