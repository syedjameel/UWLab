# Play2Perfect FurnitureBench one-leg assets

These runtime assets come from [`kushal2000/play2perfect`](https://github.com/kushal2000/play2perfect) at commit `ff2fc62873f6c6cd93164123c8358e9ce2d65c9b` and are distributed under the adjacent `LICENSE.play2perfect` (MIT).

The public Play2Perfect problem `screwing` maps to `furniture_bench.one_leg_matchedmass_sdf_hybrid_super_dense`. The source closure is retained for provenance:

- `leg/square_table_leg4_matchedmass_sdf_hybrid.urdf` and its exact visual/collision meshes;
- `table/one_leg_sdf_hybrid.urdf` and its active-hole mesh;
- reset/canonical metadata used to verify coordinate conventions.

The table URDF's mesh reference is relocated to its adjacent runtime mesh; geometry is unchanged. The source leg retains its measured `0.022750 kg` mass, center of mass, and inertia tensor.

`leg_200mm/` is the larger, long-body variant used by the DexSuite grasp-and-lift task. It keeps the source thread and 30 mm cross-section, extends only the body to make the total leg length 200 mm, and stretches the corresponding twelve convex body hulls by the same affine transform. Its `0.0573543915 kg` mass, center of mass, and inertia are recomputed at the measured source-leg density. The runtime task spawns this leg only; its support table is a generic static cuboid, not the receptive screw fixture. Play2Perfect's insertion waypoints and screw-pitch mechanics remain out of scope.
