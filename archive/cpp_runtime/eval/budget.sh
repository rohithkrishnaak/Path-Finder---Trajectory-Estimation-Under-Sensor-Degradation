#!/usr/bin/env bash
# Measure the official budgets on Linux (WSL / Docker): peak RSS (<= 8 MB) and
# compute per IMU-second (<= 20 ms), pinned to one CPU core, median of 5 runs.
#   bash src/eval/budget.sh <input.zip> [binary]
set -euo pipefail
IN=$1
BIN=${2:-$(dirname "$0")/../pathfinder}
OUT=$(mktemp)
for i in 1 2 3 4 5; do
  /usr/bin/time -f "%M %e" -o "$OUT.time" taskset -c 0 "$BIN" "$IN" -o "$OUT" --metrics "$OUT.json" 2>/dev/null
  rss_kb=$(cut -d' ' -f1 "$OUT.time")
  msps=$(python3 -c "import json;print(json.load(open('$OUT.json'))['ms_per_imu_second'])")
  echo "$rss_kb $msps"
done | sort -k2 -n | awk '{rss[NR]=$1; ms[NR]=$2} END {printf "peak RSS %.2f MB (max of runs)   compute %.3f ms per IMU-second (median)\n", (rss[1]>rss[5]?rss[1]:rss[5])/1024, ms[3]}'
rm -f "$OUT" "$OUT.time" "$OUT.json"
