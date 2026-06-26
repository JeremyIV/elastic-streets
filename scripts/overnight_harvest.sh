#!/bin/zsh
# Overnight re-harvest: replace the sparse tile-sampled congestion data for the
# four tiles cities with dense directions + map-matching speeds, then rebuild
# their gold-standard breathing maps. Resumable (both collectors skip done cells)
# and resilient (a failed step logs and skips the city).
cd /Users/jeremy/elastic-streets
PY=.venv/bin/python
LOG=/tmp/overnight_harvest.log
echo "=== overnight harvest start ($(date +%H:%M:%S)) ===" > $LOG

run () {
  city=$1; graph=$2; title=$3
  suf="_${city}_gold"
  dir="data/day_mds_${city}_dir.json"
  gold="data/day_mds_${city}_gold.json"
  echo "" >> $LOG; echo ">>>>>> $city  ($(date +%H:%M:%S))" >> $LOG

  echo "--- collect_directions ---" >> $LOG
  $PY scripts/collect_directions.py "$city" "$graph" >> $LOG 2>&1 \
    || { echo "!!! $city collect_directions FAILED" >> $LOG; return; }
  echo "--- collect_matching ---" >> $LOG
  $PY scripts/collect_matching.py "$city" "$graph" >> $LOG 2>&1 \
    || { echo "!!! $city collect_matching FAILED (continuing w/ directions only)" >> $LOG; }
  echo "--- solve_directions ---" >> $LOG
  $PY scripts/solve_directions.py "$city" "$graph" >> $LOG 2>&1 \
    || { echo "!!! $city solve_directions FAILED" >> $LOG; return; }
  echo "--- solve_sampled @2.5 ---" >> $LOG
  $PY scripts/solve_sampled.py "$dir" "$gold" --anchor 2.5 >> $LOG 2>&1 \
    || { echo "!!! $city solve_sampled FAILED" >> $LOG; return; }
  echo "--- render ---" >> $LOG
  $PY scripts/render_base.py "$gold" "$suf" >> $LOG 2>&1 \
    || { echo "!!! $city render FAILED" >> $LOG; return; }
  echo "--- compose ---" >> $LOG
  $PY scripts/compose_video.py "$suf" "$title" 1.2 >> $LOG 2>&1 \
    || { echo "!!! $city compose FAILED" >> $LOG; return; }
  echo "    OK -> shots/breathing${suf}.mp4" >> $LOG
}

run boston  data/boston_full.graphml  "Boston"
run sf       data/sf_full.graphml       "San Francisco"
run chicago  data/chicago_full.graphml  "Chicago"
run nyc      data/nyc_full.graphml      "New York"

echo "" >> $LOG
echo "=== rebuilding collection montage ===" >> $LOG
ffmpeg -y \
 -i shots/breathing_manhattan_gold.mp4 -i shots/breathing_boston_gold.mp4 \
 -i shots/breathing_sf_gold.mp4 -i shots/breathing_chicago_gold.mp4 \
 -i shots/breathing_nyc_gold.mp4 -i shots/breathing_la_gold.mp4 \
 -i shots/breathing_seattle_gold.mp4 \
 -f lavfi -t 9 -i color=c=black:s=360x600 \
 -filter_complex "[0:v]scale=360:600[a];[1:v]scale=360:600[b];[2:v]scale=360:600[c];[3:v]scale=360:600[d];[4:v]scale=360:600[e];[5:v]scale=360:600[f];[6:v]scale=360:600[g];[7:v]setsar=1[h];[a][b][c][d][e][f][g][h]xstack=inputs=8:layout=0_0|360_0|720_0|1080_0|0_600|360_600|720_600|1080_600[out]" \
 -map "[out]" -c:v libx264 -pix_fmt yuv420p -r 24 shots/breathing_gold_collection.mp4 >> $LOG 2>&1 \
 && echo "    montage OK" >> $LOG || echo "!!! montage FAILED" >> $LOG

echo "=== overnight harvest done ($(date +%H:%M:%S)) ===" >> $LOG
