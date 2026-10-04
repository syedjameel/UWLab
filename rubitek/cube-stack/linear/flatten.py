"""Flatten IsaacLab 3.0 folder-style URDF conversions into the single-file USDs the grafts expect."""
import sys
from pxr import Usd
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    stage = Usd.Stage.Open(src)
    print("FLATTEN", src, "-> default prim", stage.GetDefaultPrim().GetPath(), flush=True)
    stage.Flatten().Export(dst)
    print("WROTE", dst, flush=True)
