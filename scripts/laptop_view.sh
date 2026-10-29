#!/bin/bash
# Runs ON THE LAPTOP (WSL Ubuntu 20.04 + ROS 2 Foxy + ros-foxy-rtabmap-ros).
# Shows the live map that the Jetson is building. The Jetson must already be
# running jetson_map.sh, on the same network and the same ROS_DOMAIN_ID.
#
# NOTE: needs the laptop ROS setup (not done yet).

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}"
source /opt/ros/foxy/setup.bash

echo "Topics visible from the Jetson:"
ros2 topic list | grep -E "rtabmap|camera" | head -n 20

# rtabmap_viz subscribes to the map data the Jetson publishes (/rtabmap/mapData)
ros2 launch rtabmap_viz rtabmap_viz.launch.py
