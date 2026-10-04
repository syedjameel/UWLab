# ACTIVE RENTED INSTANCE — destroy when done

    id        49280795
    gpu       RTX A6000, 44 GB
    rate      $0.4556/h  (offer was $0.403 + storage)
    ssh       ssh2.vast.ai : 10794  (root)
    created   2026-08-31 01:43 local
    label     cubestack

## Commands
    ~/Work/Rubitek/cube-stack/vast/launch.sh status     # is it still billing?
    ~/Work/Rubitek/cube-stack/vast/launch.sh push       # rsync worktree + datasets
    ~/Work/Rubitek/cube-stack/vast/launch.sh job <f.sh> # queue a job under container init
    ~/Work/Rubitek/cube-stack/vast/launch.sh log <name>
    ~/Work/Rubitek/cube-stack/vast/launch.sh destroy    # STOPS BILLING

## Cost discipline
Credit at start: $65.76, and it is the entire runway (balance $0).
At $0.4556/h: 1 h = $0.46, 12 h = $5.47, 24 h = $10.93.
**Check `launch.sh status` at every wake-up.** An instance left running overnight by accident is
~$11, a sixth of the budget, for nothing.

## Why jobs go through the job loop, never ssh
This box family reaps detached processes -- setsid, nohup and tmux alike -- when the ssh session
that spawned them closes (recorded in meaningful-latents/LOG.md:221). `onstart.sh` therefore runs
under the container's init and polls `/workspace/jobs/*.sh`; ssh is only ever used to drop a job
file in and to read logs.

## Instance API key
The create response contained an `instance_api_key`. It has been stripped from
`vast/create.json` and that file is mode 600. Do not re-run `create` with output committed anywhere.
