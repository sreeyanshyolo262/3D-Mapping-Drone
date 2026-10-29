#!/bin/bash
# Runs ON THE JETSON with no laptop, no ssh, no internet.
# Waits for the camera, records a map for a fixed time, stops cleanly, exports the point cloud.
#
# Usage:  ./jetson_auto.sh [SECONDS] [--force]
#   SECONDS  recording time after the camera has started (default 90)
#   --force  run even if the one-shot arm file is missing (use this when starting it by hand)
#
# Started on boot by the service from install_autostart.sh. It only records if
# ~/jetson-work/.arm exists, and deletes that file as soon as it starts, so a normal boot
# on the ground does NOT record anything.
#
# Progress is written to ~/jetson-work/logs/auto_status.txt.

SECS="${1:-90}"
FORCE=0
for a in "$@"; do [ "$a" = "--force" ] && FORCE=1; done
case "$SECS" in ''|*[!0-9]*) SECS=90 ;; esac

BASE="$HOME/jetson-work"
LOG="$BASE/logs"
STATUS="$LOG/auto_status.txt"
mkdir -p "$LOG" "$BASE/maps"

say() { echo "$(date '+%F %T')  $*" | tee -a "$STATUS"; }

if [ ! -f "$BASE/.arm" ] && [ "$FORCE" -eq 0 ]; then
  say "not armed (no $BASE/.arm) - nothing to do"
  exit 0
fi
rm -f "$BASE/.arm"
touch "$BASE/.ran_now"           # tells the service a recording really happened
say "START recording for ${SECS}s"

# ROS 2 discovery needs a network interface; with no network, keep it on the loopback.
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"
sudo -n ip link set lo multicast on 2>/dev/null

# Give the USB camera time to appear after boot (up to 90 s).
for i in $(seq 1 90); do
  lsusb 2>/dev/null | grep -qi "8086" && break
  sleep 1
done
if ! lsusb 2>/dev/null | grep -qi "8086"; then
  say "ERROR camera not found on USB - aborting"
  exit 1
fi
say "camera found"

bash "$BASE/jetson_map.sh" 640x480 15 > "$LOG/auto_map.log" 2>&1 &
MAP=$!

# jetson_map.sh needs ~15 s to bring up the camera and RTAB-Map before it is mapping.
trap 'kill $SLP 2>/dev/null' INT      # Ctrl+C on a terminal = stop early and save
sleep $((SECS + 15)) &
SLP=$!
wait $SLP
say "STOP - saving and exporting (this can take a minute or more, do not cut power)"
kill -TERM "$MAP" 2>/dev/null
wait "$MAP"

LATEST=$(ls -t "$BASE"/maps/*.db 2>/dev/null | head -n 1)
if [ -n "$LATEST" ] && [ -s "$LATEST" ]; then
  say "DONE  map: $LATEST ($(du -h "$LATEST" | cut -f1))"
else
  say "ERROR no map database was written - see $LOG/auto_map.log"
fi
sync
