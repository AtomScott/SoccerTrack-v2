#!/bin/bash
# Launch one gsr4 (v2 auto-resume, fp16) instance. usage: launch_gsr4.sh <match> <half> [zone]
# The startup script is passed from a file: inline metadata cannot carry the comma in
# "remount,size=48G" (gcloud splits --metadata on commas).
set -u
export PATH=$PATH:$HOME/google-cloud-sdk/bin
M=$1; H=$2; ZN=${3:-asia-northeast1-a}
gcloud compute instances create "gsr4-$M-$H" --zone="$ZN" \
  --machine-type=g2-standard-16 \
  --image=gsr-l4-golden-fast2 \
  --boot-disk-size=300GB --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata=gsr-match=$M,gsr-half=$H,gsr-fp16=true,enable-guest-attributes=TRUE \
  --metadata-from-file=startup-script=$HOME/gsr-sweep/gsr4_startup.sh \
  2>&1 | tail -1
