# Auto-mount /mnt/hdd + auto-resume Stage 2 training on boot

Two pieces of recovery automation:

1. `/etc/fstab` already has the right entry with `nofail` — so the disk
   mounts on boot if it's present, and boot does **not** stall if it isn't.
2. **`mnt-hdd-watchdog.service`** runs a polling loop as root that:
   - re-mounts `/mnt/hdd` within ~20 s if it disappears (USB/SATA reset,
     sleep/wake, etc.)
   - after every successful mount (and at boot), checks that the **dashboard**
     and the **Stage 2 training process** are alive — relaunches them via
     `runuser -u saif` if not. This is what makes "training resumes on
     reboot" work.

The watchdog uses `ckpt_running.pth` for resume, so you lose at most the work
since the last mid-epoch save (~12 min @ `SAVE_EVERY_ITERS=2000`).

## Install

```bash
# 1. Confirm fstab is right (yours already is)
grep '/mnt/hdd' /etc/fstab
# expected: UUID=... /mnt/hdd ext4 defaults,nofail,...,x-systemd.device-timeout=10s 0 2

# 2. Install the watchdog script + service unit
sudo cp /home/saif/github/efficientsam3/scripts/systemd/mnt-hdd-watchdog.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/mnt-hdd-watchdog.sh
sudo cp /home/saif/github/efficientsam3/scripts/systemd/mnt-hdd-watchdog.service /etc/systemd/system/

# 3. Activate + start now
sudo systemctl daemon-reload
sudo systemctl enable --now mnt-hdd-watchdog

# 4. Verify
systemctl status mnt-hdd-watchdog
sudo tail -f /var/log/mnt-hdd-watchdog.log
```

You should immediately see lines like:

```
2026-05-28 01:05:12 watchdog start (mount=/mnt/hdd interval=20s)
2026-05-28 01:05:12 MOUNTED /mnt/hdd (initial)
2026-05-28 01:05:14 relaunching dashboard
2026-05-28 01:05:18 relaunching training (tag=ep50_run4)
2026-05-28 01:05:20   training pid=...
```

If the watchdog finds training + dashboard already alive (your case right
now), it just stays quiet and polls.

## Tests

**Simulated runtime unmount** — confirms remount + auto-relaunch within ~30 s:
```bash
sudo umount /mnt/hdd
# wait 25–30 s
mountpoint /mnt/hdd && tail -5 /var/log/mnt-hdd-watchdog.log
```

**Simulated training crash** — confirms relaunch on next 20-second tick:
```bash
sudo pkill -KILL -f "stage2/train.py"
# wait 25–30 s
pgrep -af stage2/train.py
```

**Full reboot test** — the real thing:
```bash
sudo reboot
# after the machine comes back up, log back in and check:
systemctl status mnt-hdd-watchdog
mountpoint /mnt/hdd
pgrep -af stage2/train.py
pgrep -af "uvicorn dashboard"
tail /var/log/mnt-hdd-watchdog.log
```

## Uninstall

```bash
sudo systemctl disable --now mnt-hdd-watchdog
sudo rm /etc/systemd/system/mnt-hdd-watchdog.service /usr/local/bin/mnt-hdd-watchdog.sh
```

## Notes / gotchas

- The watchdog will keep relaunching training on every crash. If you want it
  off temporarily (e.g. to do code edits), `sudo systemctl stop
  mnt-hdd-watchdog` first; remember to `sudo systemctl start
  mnt-hdd-watchdog` when you're done.
- It only relaunches if `logs/stage2_run4.pid` is dead. If training is
  legitimately stopped because you killed it manually and want it to **stay**
  stopped, also stop the watchdog (otherwise it'll resurrect training in
  ~20 s).
- Tag is hard-coded to `ep50_run4`. Change `TAG=…` at the top of
  `mnt-hdd-watchdog.sh` for future runs and re-copy.
- The watchdog runs as root but uses `runuser -u saif` to drop privileges
  before launching torchrun / uvicorn. The Python processes themselves run
  as user `saif`, so they have correct CUDA / venv access.
- Mount uses `mount /mnt/hdd` (reads fstab) — stays in sync with whatever
  UUID / options are configured there, no hard-coded device path.
