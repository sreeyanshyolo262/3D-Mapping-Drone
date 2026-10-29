#!/bin/bash
# Runs ON THE JETSON. Starts the D455 + RTAB-Map, builds a 3D map while you move
# the camera, and on Ctrl+C saves the map database and exports a point cloud (.ply).
#
# Usage:  ./jetson_map.sh [WIDTHxHEIGHT] [FPS]      e.g. ./jetson_map.sh 640x480 15
#
# Output: ~/jetson-work/maps/map_<time>.db   (open with rtabmap-databaseViewer)
#         ~/jetson-work/maps/map_<time>_cloud.ply  (open in MeshLab / CloudCompare)

RES="${1:-640x480}"
FPS="${2:-15}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"

source /opt/ros/foxy/setup.bash 2>/dev/null
source "$HOME/ros2_ws/install/setup.bash" 2>/dev/null

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$HOME/jetson-work/maps"
LOG="$HOME/jetson-work/logs"
mkdir -p "$OUT" "$LOG"
DB="$OUT/map_$STAMP.db"

PIDS=()
cleanup() {
  trap - INT TERM EXIT
  echo; echo "[map] stopping..."
  for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
  sleep 4
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  pkill -f realsense2_camera_node 2>/dev/null
  sleep 1
  if [ -s "$DB" ]; then
    echo "[map] saved database: $DB"
    # RTAB-Map sometimes writes a lone, unconnected frame at shutdown, and the exporter then
    # exports only that one frame. Export from a repaired copy when that happens.
    HERE="$(cd "$(dirname "$0")" && pwd)"
    EXPORT_DB="$DB"
    FIXED="${DB%.db}_fixed.db"
    if [ -f "$HERE/fix_last_node.py" ] && python3 "$HERE/fix_last_node.py" "$DB" "$FIXED"; then
      EXPORT_DB="$FIXED"
    fi
    echo "[map] exporting point cloud (this can take a minute)..."
    EXPORT_ARGS="--cloud --decimation 1 --voxel 0.01 --max_range 9 --noise_radius 0.05 --noise_k 5 --output map_$STAMP"
    (cd "$OUT" && rtabmap-export $EXPORT_ARGS "$EXPORT_DB" >/dev/null 2>&1)
    if [ ! -s "$OUT/map_${STAMP}_cloud.ply" ] && [ "$EXPORT_DB" != "$DB" ]; then
      echo "[map] export from the repaired copy failed; exporting the original database instead"
      (cd "$OUT" && rtabmap-export $EXPORT_ARGS "$DB" >/dev/null 2>&1)
    fi
    rm -f "$FIXED"
    ls -1 "$OUT"/map_${STAMP}* 2>/dev/null
  else
    echo "[map] no database was written (map empty?). See $LOG/rtabmap_$STAMP.log"
  fi
  exit 0
}
trap cleanup INT TERM

# Stop any camera node left over from earlier tests (own processes only).
pkill -f realsense2_camera_node 2>/dev/null && sleep 2

echo "[map] starting camera at $RES @ ${FPS}fps ..."
ros2 launch realsense2_camera rs_launch.py \
  depth_module.profile:=${RES}x${FPS} \
  rgb_camera.profile:=${RES}x${FPS} \
  align_depth.enable:=true \
  > "$LOG/camera_$STAMP.log" 2>&1 &
PIDS+=($!)
sleep 8

echo "[map] starting RTAB-Map ..."
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/camera/color/image_raw \
  depth_topic:=/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/color/camera_info \
  frame_id:=camera_link \
  approx_sync:=true \
  qos:=2 \
  rtabmap_viz:=false \
  rviz:=false \
  database_path:="$DB" \
  rtabmap_args:="--delete_db_on_start --Vis/MaxFeatures 600 --Kp/MaxFeatures 400 --Rtabmap/DetectionRate 2 --RGBD/LinearUpdate 0.05 --RGBD/AngularUpdate 0.05" \
  > "$LOG/rtabmap_$STAMP.log" 2>&1 &
PIDS+=($!)

echo "[map] RUNNING. Move the camera slowly. Ctrl+C to stop and save."
echo "[map] logs: $LOG/*_$STAMP.log"
while true; do
  sleep 5
  if [ -f "$DB" ]; then printf "[map] db size: %s\r" "$(du -h "$DB" | cut -f1)"; fi
done
