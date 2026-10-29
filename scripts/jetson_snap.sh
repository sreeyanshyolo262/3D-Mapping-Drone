#!/bin/bash
# Runs ON THE JETSON (its own terminal, no network needed).
# Stop-and-snap capture for a walk-through map.
#
# Usage: ./jetson_snap.sh [AUTO_SECONDS] [WIDTHxHEIGHT] [FPS]
#   ./jetson_snap.sh        press Enter at every stop to take a snap
#   ./jetson_snap.sh 6      snaps by itself every 6 s (with a 3-2-1 countdown)
# Output: ~/jetson-work/snaps/snaps_<time>/   (copy to the laptop with build_streetview.ps1)

AUTO="${1:-0}"
RES="${2:-640x480}"
FPS="${3:-10}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"
# ROS needs loopback multicast to work with no network cable. Turn it on if we can; if we
# can't (no sudo rights), don't force loopback-only, so it still works with the cable plugged in.
sudo -n ip link set lo multicast on 2>/dev/null
if ip link show lo | head -n 1 | grep -q MULTICAST; then
  export ROS_LOCALHOST_ONLY=1
else
  unset ROS_LOCALHOST_ONLY
  echo "[snap] NOTE: loopback multicast is off, so keep the ethernet cable plugged in while snapping"
  echo "[snap]       (or run once:  sudo ip link set lo multicast on   to work without a cable)"
fi

source /opt/ros/foxy/setup.bash 2>/dev/null
source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HOME/jetson-work/logs"
SESSION="$HOME/jetson-work/snaps/snaps_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG" "$SESSION"

cleanup() {
  trap - INT TERM HUP EXIT
  pkill -f "[r]ealsense2_camera_nod[e]" 2>/dev/null
}
trap cleanup EXIT HUP TERM

pkill -f "[r]ealsense2_camera_nod[e]" 2>/dev/null && sleep 2

echo "[snap] starting camera at $RES @ ${FPS}fps ..."
ros2 launch realsense2_camera rs_launch.py \
  depth_module.profile:=${RES}x${FPS} \
  rgb_camera.profile:=${RES}x${FPS} \
  align_depth.enable:=true \
  > "$LOG/snap_camera.log" 2>&1 &
sleep 9

python3 "$HERE/jetson_snap.py" "$SESSION" "$AUTO"
echo "[snap] session folder: $SESSION"
