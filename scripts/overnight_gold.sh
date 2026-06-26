#!/bin/zsh
# Overnight gold-standard batch: solve_sampled @ anchor 2.5 -> render -> compose
# for every non-Uber city layout. Sequential so big all-pairs Dijkstras never
# overlap in memory. Resilient: a failed city is logged and skipped.
cd /Users/jeremy/elastic-streets
PY=.venv/bin/python
LOG=/tmp/overnight_gold.log
echo "=== overnight gold batch start ===" > $LOG

# city  input_json                       title
run () {
  city=$1; inp=$2; title=$3
  suf="_${city}_gold"
  out="data/day_mds_${city}_gold.json"
  echo "" >> $LOG
  echo ">>> $city  ($(date +%H:%M:%S))" >> $LOG
  if [ ! -f "$out" ]; then
    $PY scripts/solve_sampled.py "$inp" "$out" --anchor 2.5 >> $LOG 2>&1 \
      || { echo "!!! $city SOLVE FAILED" >> $LOG; return; }
  else
    echo "    (layout exists, skipping solve)" >> $LOG
  fi
  $PY scripts/render_base.py "$out" "$suf" >> $LOG 2>&1 \
    || { echo "!!! $city RENDER FAILED" >> $LOG; return; }
  $PY scripts/compose_video.py "$suf" "$title" 1.2 >> $LOG 2>&1 \
    || { echo "!!! $city COMPOSE FAILED" >> $LOG; return; }
  echo "    OK -> shots/breathing${suf}.mp4" >> $LOG
}

run manhattan data/day_mds_manhattan_dir.json "Manhattan"
run boston    data/day_mds_boston.json        "Boston"
run sf        data/day_mds_sf.json             "San Francisco"
run chicago   data/day_mds_chicago.json        "Chicago"
run nyc       data/day_mds_nyc.json            "New York"
run la        data/day_mds_la_full.json        "Los Angeles"
run seattle   data/day_mds_seattle_dir.json    "Seattle"

echo "" >> $LOG
echo "=== overnight gold batch done ($(date +%H:%M:%S)) ===" >> $LOG
ls -la shots/breathing_*_gold.mp4 >> $LOG 2>&1
