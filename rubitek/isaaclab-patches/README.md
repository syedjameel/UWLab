Isaac Sim 6 port of IsaacLab used by this branch: 2 commits on top of upstream IsaacLab
`99f1423` ("Install docs dependencies for beta2 multi-version build (#7578)", branch `release/3.0.0-beta2`).

    git clone --branch release/3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git _isaaclab/IsaacLab
    git -C _isaaclab/IsaacLab checkout 99f1423
    git -C _isaaclab/IsaacLab am ../../rubitek/isaaclab-patches/*.patch
