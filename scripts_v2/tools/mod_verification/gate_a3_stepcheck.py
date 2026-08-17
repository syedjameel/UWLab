"""Phase A3 behavioural gate for the 5090 (bead UWLab-qiao.1 5090-migration follow-on).

Minimal AppLauncher -> SimulationContext -> reset -> 10 steps check. A version string printing
successfully is NOT proof the GPU sim path works -- this actually steps physics on cuda:0.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

sim_cfg = SimulationCfg(dt=1.0 / 60.0, device="cuda:0")
sim = SimulationContext(sim_cfg)
sim.reset()
print("SIM_RESET_OK", flush=True)

for i in range(10):
    sim.step()
    print(f"step {i} ok", flush=True)

print("STEPPED_10_OK", flush=True)
simulation_app.close()
