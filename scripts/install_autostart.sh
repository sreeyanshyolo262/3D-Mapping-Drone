#!/bin/bash
# Runs ON THE JETSON, once (needs sudo). Installs a boot service that runs jetson_auto.sh.
#
# Usage:  sudo ./install_autostart.sh [SECONDS] [shutdown]
#   SECONDS   recording time (default 90)
#   shutdown  power the Jetson off by itself after the map is saved
#
# After installing, nothing records on a normal boot. To record once on the next boot:
#   touch ~/jetson-work/.arm
# Remove it again (service stays installed):
#   sudo systemctl disable drone-map

set -e
SECS="${1:-90}"
USER_NAME="${SUDO_USER:-$USER}"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
BASE="$USER_HOME/jetson-work"

if [ ! -f "$BASE/jetson_auto.sh" ] || [ ! -f "$BASE/jetson_map.sh" ]; then
  echo "Copy jetson_auto.sh and jetson_map.sh to $BASE first."
  exit 1
fi
chmod +x "$BASE/jetson_auto.sh" "$BASE/jetson_map.sh"

if [ "$2" = "shutdown" ]; then touch "$BASE/.shutdown_after"; else rm -f "$BASE/.shutdown_after"; fi
chown "$USER_NAME" "$BASE/.shutdown_after" 2>/dev/null || true

cat > /etc/systemd/system/drone-map.service <<EOF
[Unit]
Description=Drone depth mapping (records once when armed)
After=multi-user.target

[Service]
Type=simple
User=$USER_NAME
Environment=HOME=$USER_HOME
ExecStartPre=+/sbin/ip link set lo multicast on
ExecStart=/bin/bash $BASE/jetson_auto.sh $SECS
ExecStopPost=+/bin/bash -c '[ -f $BASE/.ran_now ] && rm -f $BASE/.ran_now && [ -f $BASE/.shutdown_after ] && /sbin/shutdown -h now || true'
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable drone-map.service
echo
echo "Installed. Recording time: ${SECS}s. Auto power-off: $([ -f "$BASE/.shutdown_after" ] && echo yes || echo no)"
echo "Arm the next boot with:   touch $BASE/.arm"
echo "Check what happened with: cat $BASE/logs/auto_status.txt"
