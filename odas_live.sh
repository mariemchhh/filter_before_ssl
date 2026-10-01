#!/usr/bin/env bash
# odas_live.sh - clean (re)start of ODAS live for any array (UMA-16, UMA-8): kill old, free the card, server, core, check /sst.
# Usage: bash odas_live.sh [cfg=~/Downloads/uma16.cfg]
# Stop everything later with:  bash odas_live.sh stop
CFG=${1:-$HOME/Downloads/uma16.cfg}
LOG=~/dataset/odas_logs; mkdir -p "$LOG"
source ~/ros2_ws/install/setup.bash   # must be sourced without 'set -u' (colcon uses unset vars)
set -u

stop_all() {
  pkill -9 -x odas_core_node 2>/dev/null
  pkill -9 -x odaslive 2>/dev/null
  pkill -9 -f odas_core_node.launch 2>/dev/null
  pkill -9 -f odas_server_node 2>/dev/null
  pkill -9 -f odas_visualization_node 2>/dev/null
  pkill -9 -f "odas.launch.xml" 2>/dev/null
  pkill -9 -x arecord 2>/dev/null
  sleep 1
}

if [ "$CFG" = "stop" ]; then stop_all; echo "ODAS stopped."; exit 0; fi

echo "[1/5] stopping old ODAS processes"
stop_all

echo "[2/5] checking the microphone array"
# card number: "card = N", or devicename "hw:N,0" / "plughw:N,0" / "plughw:CARD=NAME,DEV=0"
CARD=$(grep -oP '^\s*card\s*=\s*\K[0-9]+' "$CFG" | head -1)
[ -z "$CARD" ] && CARD=$(grep -oP 'devicename\s*=\s*"(plug)?hw:\K[0-9]+' "$CFG" | head -1)
if [ -z "$CARD" ]; then
  NAME=$(grep -oP 'devicename\s*=\s*"(plug)?hw:CARD=\K[^,"]+' "$CFG" | head -1)
  [ -n "$NAME" ] && CARD=$(readlink /proc/asound/"$NAME" 2>/dev/null | grep -oP 'card\K[0-9]+')
fi
CARD=${CARD:-2}
echo "  cfg uses card $CARD: $(arecord -l | grep "card $CARD:" | cut -d: -f2 | xargs)"
if arecord -l | grep "card $CARD:" | grep -qiE "HDA Intel|PCH|ALC[0-9]"; then
  echo "  [FAIL] card $CARD is the laptop's built-in mic, not a mic array. Fix 'devicename' / 'card' in $CFG:"
  arecord -l | grep "^card"; exit 1
fi
if ! arecord -l | grep -q "card $CARD:"; then
  echo "  [FAIL] card $CARD not found. Current cards:"; arecord -l; exit 1
fi
HOLD=$(fuser /dev/snd/pcmC${CARD}D0c 2>/dev/null)
if [ -n "$HOLD" ]; then
  echo "  card held by PID(s):$HOLD -> killing"; kill -9 $HOLD 2>/dev/null; sleep 1
fi
fuser /dev/snd/pcmC${CARD}D0c >/dev/null 2>&1 && { echo "  [FAIL] card still busy:"; fuser -v /dev/snd/pcmC${CARD}D0c; exit 1; }
echo "  card $CARD free"

echo "[3/5] starting server (log: $LOG/server.log)"
nohup ros2 launch odas_ros odas.launch.xml configuration_path:="$CFG" > "$LOG/server.log" 2>&1 &
# ports = the socket sinks declared in this cfg (UMA-16 cfg: 9000-9002, a UMA-8 cfg may have fewer)
PORTS=$(grep -oP '^\s*port\s*=\s*\K[0-9]+' "$CFG" | sort -u | xargs)
[ -z "$PORTS" ] && PORTS="9000 9001"
listening() { local n=0; for p in $PORTS; do ss -ltn | grep -q ":$p " && n=$((n+1)); done; echo $n; }
NP=$(echo $PORTS | wc -w)
for i in $(seq 1 30); do [ "$(listening)" -ge "$NP" ] && break; sleep 0.5; done
if [ "$(listening)" -lt "$NP" ]; then
  echo "  [FAIL] server not listening on all cfg ports ($PORTS)"; ss -ltn | grep -E ':90[0-9]{2} '; tail -20 "$LOG/server.log"; exit 1
fi
echo "  server listening on $PORTS"
sleep 2

echo "[4/5] starting core (log: $LOG/core.log)"
nohup pasuspender -- ros2 run odas_ros odas_core_node --ros-args -p configuration_path:="$CFG" > "$LOG/core.log" 2>&1 &
sleep 4
if ! pgrep -x odas_core_node >/dev/null; then
  echo "  [FAIL] core died. Last lines:"; tail -15 "$LOG/core.log"; exit 1
fi
echo "  core running (PID $(pgrep -x odas_core_node))"

echo "[5/5] checking /sst"
if timeout 6 ros2 topic echo /sst --once >/dev/null 2>&1; then
  echo "  [OK] /sst is publishing. ODAS is live."
  echo
  if echo "$CFG" | grep -qi uma8; then
    echo "Next:  python3 ~/Downloads/uma8_known_track_drop_node.py --ros-args -p calib_seconds:=10.0 -p min_activity:=0.05"
    echo "       rviz2 -d ~/Downloads/uma8_view.rviz"
  else
    echo "Next:  python3 ~/Downloads/known_track_drop_node.py --ros-args -p known_az:=21.2 -p known_el:=-8.3 -p vertical:=true -p min_activity:=0.05 -p label_mode:=id"
    echo "       rviz2 -d ~/Downloads/target_view.rviz"
  fi
  echo "Logs:  tail -f $LOG/core.log   |   stop: bash odas_live.sh stop"
else
  echo "  [FAIL] no /sst message. Core log:"; tail -15 "$LOG/core.log"; exit 1
fi
