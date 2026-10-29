#!/bin/bash
# Runs ON THE JETSON. Starts the D455, saves ONE colour + ONE depth frame as PNGs, stops the camera.
# Usage: ./jetson_capture.sh [PREFIX] [WIDTHxHEIGHT] [FPS]
# Output: ~/jetson-work/shots/<PREFIX>_color.png, _depth.png, _depth_filled.png, _depth_raw16.png

PREFIX="${1:-shot}"
RES="${2:-640x480}"
FPS="${3:-10}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"

source /opt/ros/foxy/setup.bash 2>/dev/null
source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HOME/jetson-work/logs"
OUT="$HOME/jetson-work/shots"
mkdir -p "$LOG" "$OUT"

cleanup() {
  trap - INT TERM HUP EXIT
  pkill -f "[r]ealsense2_camera_nod[e]" 2>/dev/null
}
trap cleanup INT TERM HUP EXIT

pkill -f "[r]ealsense2_camera_nod[e]" 2>/dev/null && sleep 2

ros2 launch realsense2_camera rs_launch.py \
  depth_module.profile:=${RES}x${FPS} \
  rgb_camera.profile:=${RES}x${FPS} \
  align_depth.enable:=true \
  > "$LOG/capture_camera.log" 2>&1 &
sleep 9

python3 "$HERE/jetson_capture.py" "$OUT/$PREFIX" 3
echo "[capture] corrupt frames in log: $(grep -c 'Incomplete video frame' "$LOG/capture_camera.log")"
