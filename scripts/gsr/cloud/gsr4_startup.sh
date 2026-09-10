#!/bin/bash
# gsr4 (v2 auto-resume, fp16) boot: shm/swap mitigation, then the v2 runner.
mount -o remount,size=48G /dev/shm
if ! swapon --show 2>/dev/null | grep -q .; then
  fallocate -l 24G /swapfile && chmod 600 /swapfile && mkswap -f /swapfile && swapon /swapfile
fi
sudo -u atom bash -c "nohup /home/atom/run_half_v2.sh >/dev/null 2>&1 &"
