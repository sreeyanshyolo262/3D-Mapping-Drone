#!/bin/bash
# Runs ON THE JETSON. Starts the D455 and a live web preview (color | depth).
# Usage: ./jetson_preview.sh [COLOR_WxH] [DEPTH_WxH] [FPS]
#   e.g. ./jetson_preview.sh 1280x720 640x480 15
# Colour is the sharp one (RGB sensor goes up to 1280x800); depth is kept at 640x480 so the
# USB-powered board isn't overloaded. Then open http://<jetson-ip>:8080. Ctrl+C stops everything.

CRES="${1:-640x480}"       # defaults match what the USB 2 link can carry
DRES="${2:-640x480}"
FPS="${3:-10}"
CW="${CRES%x*}"; CH="${CRES#*x}"
DW="${DRES%x*}"; DH="${DRES#*x}"
PORT=8080
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"

source /opt/ros/foxy/setup.bash 2>/dev/null
source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null

LOG="$HOME/jetson-work/logs"
mkdir -p "$LOG"
HERE="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  trap - INT TERM HUP EXIT
  echo; echo "[preview] stopping..."
  pkill -f realsense2_camera_node 2>/dev/null
  pkill -f jetson_preview.py 2>/dev/null
  exit 0
}
trap cleanup INT TERM HUP EXIT

# Stop any camera node left over from an earlier run (it can only be opened once).
pkill -f realsense2_camera_node 2>/dev/null && sleep 2

echo "[preview] starting camera: color $CRES, depth $DRES @ ${FPS}fps ..."
ros2 launch realsense2_camera rs_launch.py \
  depth_module.profile:=${DRES}x${FPS} \
  rgb_camera.profile:=${CRES}x${FPS} \
  > "$LOG/preview_camera.log" 2>&1 &
sleep 8

IP=$(hostname -I | awk '{print $1}')
echo "[preview] OPEN IN BROWSER:  http://$IP:$PORT"
echo "[preview] Ctrl+C to stop."
# Run in the background and `wait`, so the trap fires immediately on Ctrl+C / SSH drop.
python3 "$HERE/jetson_preview.py" "$CW" "$CH" "$DW" "$DH" "$PORT" &
wait $!
