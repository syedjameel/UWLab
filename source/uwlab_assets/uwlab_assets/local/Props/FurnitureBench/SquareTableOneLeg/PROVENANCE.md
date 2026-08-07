# Play2Perfect FurnitureBench one-leg assets

These runtime assets come from [`kushal2000/play2perfect`](https://github.com/kushal2000/play2perfect) at commit `ff2fc62873f6c6cd93164123c8358e9ce2d65c9b` and are distributed under the adjacent `LICENSE.play2perfect` (MIT).

The public Play2Perfect problem `screwing` maps to `furniture_bench.one_leg_matchedmass_sdf_hybrid_super_dense`. This directory retains only that problem's grasped leg and receptive table closure:

- `leg/square_table_leg4_matchedmass_sdf_hybrid.urdf` and its exact visual/collision meshes;
- `table/one_leg_sdf_hybrid.urdf` and its active-hole mesh;
- reset/canonical metadata used to verify coordinate conventions.

The table URDF's mesh reference is relocated to its adjacent runtime mesh; geometry is unchanged. The leg retains its measured `0.022750 kg` mass, center of mass, and inertia tensor. OmniReset uses these assets for grasp and lift only; Play2Perfect's insertion waypoints and screw-pitch mechanics are intentionally out of scope.
