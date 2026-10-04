# ACTIVE RENTED INSTANCE (Isaac Sim 6) — destroy when done

    id        51171924
    offer     50629593
    gpu       RTX 4090, 23 GB VRAM, 62 GB RAM, 32 vCPU, 633 GB disk
    rate      $0.455/h
    driver    580.159.03  (the 580 branch the RTX renderer needs; 595 segfaults rtx.scenedb)
    account   SINEW (Rubitek / UWLab work, id 636218) — key ~/.config/vastai/keys/sinew
    created   2026-09-16 ~10:5x local
    purpose   scale the DELTO pinch reset banks, then run the staged training curriculum

## Commands
    cube-stack/vast/launch_is6.sh status              # is it up / still billing?
    cube-stack/vast/launch_is6.sh push                # worktree + _isaaclab3 + compat + datasets
    cube-stack/vast/launch_is6.sh job <f.sh>          # queue a job under container init
    cube-stack/vast/launch_is6.sh log <name>          # tail a job log
    cube-stack/vast/launch_is6.sh pull <remote> <local>
    cube-stack/vast/launch_is6.sh destroy             # STOPS BILLING

## Cost discipline
$0.455/h → 1 h $0.46, 12 h $5.5, 24 h $10.9. Sinew credit was $410.67 but is shared with other
Rubitek work (3 unrelated RTX 5090s were already running). Budget for this box: ~$10, i.e. destroy
within about a day. Check `status` at every wake-up; an idle rented box is pure loss.

**Never destroy an instance this script did not create** — only the id in `vast/instance_id_is6`.

## Limitations found on this box
- **No RTX rendering.** The container runs with `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so
  there is no NVIDIA Vulkan ICD/producer and `vulkaninfo` resolves to llvmpipe; every viewport render
  dies with "vkCreateInstance failed. Vulkan 1.1 is not supported". The capability is fixed at
  container creation, so it cannot be repaired in place. Consequences: bank QA on the box uses
  `validate_grasped_bank.py` (physics + fingertip contact, no rendering), and any FILM is rendered
  locally from a pulled checkpoint. A future box needs the graphics capability at create time.
- **`uwlab_is6_bridge` had a hardcoded dev-box path** to `isaacsim/extsDeprecated`, so `add_path`
  silently registered nothing here and every record/train job died with
  "No module named 'isaacsim.core.utils'". Fixed to resolve from the installed `isaacsim` package
  (override: `UWLAB_IS6_DEPRECATED_EXT_DIR`).

## Notes
- Team accounts reject account-level SSH keys, so `onstart_is6.sh` injects `~/.ssh/id_ed25519.pub`
  into the container's authorized_keys, and only the `--direct` endpoint works.
- `create_is6.json` had its `instance_api_key` stripped and is mode 600. Do not re-create with the
  raw response committed anywhere.
- Jobs must go through the job loop, never a bare ssh command: this box family reaps detached
  processes (setsid/nohup/tmux) when the ssh session that started them closes.
