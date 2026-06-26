#!/bin/zsh
# Queued LA re-harvest: switch LA from the anchor (point-to-point) speed source
# to the directions + map-matching source the other cities use. Waits for the
# main overnight_harvest batch to finish first (avoid concurrent API/memory load),
# then runs LA and rebuilds the final collection montage.
cd /Users/jeremy/elastic-streets
PY=.venv/bin/python
LOG=/tmp/overnight_la.log
echo "=== LA harvest queued, waiting for main batch ($(date +%H:%M:%S)) ===" > $LOG

# wait for the main harvest to exit
while pgrep -f overnight_harvest.sh >/dev/null; do sleep 60; done
echo "=== main batch done; starting LA ($(date +%H:%M:%S)) ===" >> $LOG

city=la; graph=data/la_full.graphml; title="Los Angeles"
suf="_la_gold"; dir="data/day_mds_la_dir.json"; gold="data/day_mds_la_gold.json"

echo "--- collect_directions ---" >> $LOG
$PY scripts/collect_directions.py "$city" "$graph" >> $LOG 2>&1 \
  || { echo "!!! la collect_directions FAILED" >> $LOG; exit 1; }
echo "--- collect_matching ---" >> $LOG
$PY scripts/collect_matching.py "$city" "$graph" >> $LOG 2>&1 \
  || echo "!!! la collect_matching FAILED (continuing directions-only)" >> $LOG
echo "--- solve_directions ---" >> $LOG
$PY scripts/solve_directions.py "$city" "$graph" >> $LOG 2>&1 \
  || { echo "!!! la solve_directions FAILED" >> $LOG; exit 1; }
echo "--- solve_sampled @2.5 ---" >> $LOG
$PY scripts/solve_sampled.py "$dir" "$gold" --anchor 2.5 >> $LOG 2>&1 \
  || { echo "!!! la solve_sampled FAILED" >> $LOG; exit 1; }
echo "--- render ---" >> $LOG
$PY scripts/render_base.py "$gold" "$suf" >> $LOG 2>&1 \
  || { echo "!!! la render FAILED" >> $LOG; exit 1; }
echo "--- compose ---" >> $LOG
$PY scripts/compose_video.py "$suf" "$title" 1.2 >> $LOG 2>&1 \
  || { echo "!!! la compose FAILED" >> $LOG; exit 1; }
echo "    OK -> shots/breathing_la_gold.mp4" >> $LOG

echo "=== rebuilding collection montage (with new LA) ===" >> $LOG
ffmpeg -y \
 -i shots/breathing_manhattan_gold.mp4 -i shots/breathing_boston_gold.mp4 \
 -i shots/breathing_sf_gold.mp4 -i shots/breathing_chicago_gold.mp4 \
 -i shots/breathing_nyc_gold.mp4 -i shots/breathing_la_gold.mp4 \
 -i shots/breathing_seattle_gold.mp4 \
 -f lavfi -t 9 -i color=c=black:s=360x600 \
 -filter_complex "[0:v]scale=360:600[a];[1:v]scale=360:600[b];[2:v]scale=360:600[c];[3:v]scale=360:600[d];[4:v]scale=360:600[e];[5:v]scale=360:600[f];[6:v]scale=360:600[g];[7:v]setsar=1[h];[a][b][c][d][e][f][g][h]xstack=inputs=8:layout=0_0|360_0|720_0|1080_0|0_600|360_600|720_600|1080_600[out]" \
 -map "[out]" -c:v libx264 -pix_fmt yuv420p -r 24 shots/breathing_gold_collection.mp4 >> $LOG 2>&1 \
 && echo "    montage OK" >> $LOG || echo "!!! montage FAILED" >> $LOG

echo "=== LA harvest done ($(date +%H:%M:%S)) ===" >> $LOG
