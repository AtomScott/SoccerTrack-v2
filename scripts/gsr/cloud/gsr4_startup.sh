#!/bin/bash
# gsr4 (v2 auto-resume, fp16) boot: keep atom's POSIX shared memory alive across SSH logouts
# (systemd-logind RemoveIPC=yes deletes the user's /dev/shm objects when the last session
# ends -- this is what aborted the pytorch DataLoader runs whenever an operator logged in),
# then the shm/swap mitigation, then the v2 runner.
loginctl enable-linger atom
sed -i 's/^#\?RemoveIPC=.*/RemoveIPC=no/' /etc/systemd/logind.conf
systemctl restart systemd-logind
mount -o remount,size=48G /dev/shm
if ! swapon --show 2>/dev/null | grep -q .; then
  fallocate -l 24G /swapfile && chmod 600 /swapfile && mkswap -f /swapfile && swapon /swapfile
fi
sudo -u atom bash -c "nohup /home/atom/run_half_v2.sh >/dev/null 2>&1 &"
