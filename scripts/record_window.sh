#!/usr/bin/env bash
# Record a screen region (a browser window) with ffmpeg x11grab.
#
# Usage:
#   record_window.sh select              # click the target window, cache geometry
#   record_window.sh start <output.mp4>  # start recording in background, print PID
#   record_window.sh stop                # stop recording (uses cached PID)
#   record_window.sh status              # show cached geometry + running PID
#
# Geometry cache: /tmp/vt_demo_window.geom  (W H X Y)
# PID cache:      /tmp/vt_demo_ffmpeg.pid
set -euo pipefail

GEOM_FILE=/tmp/vt_demo_window.geom
PID_FILE=/tmp/vt_demo_ffmpeg.pid
DISPLAY="${DISPLAY:-:1}"
export DISPLAY

cmd="${1:-}"
case "$cmd" in
  select)
    delay="${2:-5}"
    echo "You have ${delay}s to alt-tab to the browser window..."
    for i in $(seq "$delay" -1 1); do
      printf "  %ss...\r" "$i"
      sleep 1
    done
    echo "Now click the target browser window.        "
    # xwininfo interactively asks the user to click a window
    info=$(xwininfo)
    W=$(echo "$info" | awk '/Width:/  {print $2}')
    H=$(echo "$info" | awk '/Height:/ {print $2}')
    X=$(echo "$info" | awk '/Absolute upper-left X:/ {print $4}')
    Y=$(echo "$info" | awk '/Absolute upper-left Y:/ {print $4}')
    # Force even width/height (H.264 requirement)
    W=$(( W - (W % 2) ))
    H=$(( H - (H % 2) ))
    echo "$W $H $X $Y" > "$GEOM_FILE"
    echo "Captured: ${W}x${H}+${X}+${Y}  -> $GEOM_FILE"
    ;;
  start)
    out="${2:?usage: start <output.mp4>}"
    if [[ ! -f "$GEOM_FILE" ]]; then
      echo "No geometry cached. Run: $0 select" >&2
      exit 1
    fi
    read -r W H X Y < "$GEOM_FILE"
    mkdir -p "$(dirname "$out")"
    echo "Recording ${W}x${H}+${X}+${Y} -> $out"
    # -draw_mouse 1 keeps the cursor visible (useful for showing clicks)
    # -framerate 30 balances size vs smoothness
    # H.264 yuv420p for broad compatibility; CRF 23 ~ good quality/size
    nohup ffmpeg -y -hide_banner -loglevel warning \
        -f x11grab -framerate 30 -video_size "${W}x${H}" -i "${DISPLAY}.0+${X},${Y}" \
        -vf "format=yuv420p" \
        -c:v libx264 -preset veryfast -crf 23 \
        "$out" > /tmp/vt_demo_ffmpeg.log 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started PID=$(cat "$PID_FILE")  log=/tmp/vt_demo_ffmpeg.log"
    ;;
  stop)
    if [[ ! -f "$PID_FILE" ]]; then
      echo "No PID cached." >&2
      exit 1
    fi
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      # SIGINT so ffmpeg finalizes the MP4 moov atom
      kill -INT "$pid"
      # Wait for ffmpeg to finish flushing
      for _ in {1..30}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
      echo "Stopped PID=$pid"
    else
      echo "PID $pid not running"
    fi
    rm -f "$PID_FILE"
    ;;
  status)
    if [[ -f "$GEOM_FILE" ]]; then
      echo "Geometry: $(cat "$GEOM_FILE")  ($GEOM_FILE)"
    else
      echo "Geometry: <none>"
    fi
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "ffmpeg:   running PID=$(cat "$PID_FILE")"
    else
      echo "ffmpeg:   not running"
    fi
    ;;
  *)
    sed -n '1,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
