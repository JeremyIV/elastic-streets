#!/bin/zsh
# FULL recompute of every city in data/cities.json through the golden pipeline,
# run as a PARALLEL POOL (MAXJOBS at a time) since each solve is ~35-40 min and
# the box has the cores + RAM. Per city: fetch graph -> collect_directions ->
# collect_matching -> solve_directions (data-prep) -> solve_sampled --anchor 2.5
# --shape 0.1 -> render -> compose. Quality identical to sequential; just overlapped.
#
# Resume-safe: each city claimed via an atomic `mkdir` lock (released on exit),
# success recorded by data/.recompute_done_<city>; harvest collectors skip done
# cells. Re-running skips finished cities and retries failed ones. Order puts the
# already-harvested cities first (no API) so they land soonest.
cd /Users/jeremy/elastic-streets
PY=.venv/bin/python
LOG=/tmp/recompute_all.log
MAXJOBS=3
[ -f $LOG ] || echo "=== full recompute start $(date) ===" > $LOG
echo "=== (re)launch $(date), MAXJOBS=$MAXJOBS ===" >> $LOG

CITIES=(
  "boston:Boston" "sf:San Francisco" "chicago:Chicago" "nyc:New York" "la:Los Angeles"
  "seattle:Seattle"
  "dc:Washington DC" "portland:Portland" "vancouver:Vancouver" "toronto:Toronto"
  "atlanta:Atlanta" "austin:Austin" "houston:Houston" "dallas:Dallas"
  "bogota:Bogota" "saopaulo:Sao Paulo"
)

process_city () {
  entry=$1
  city=${entry%%:*}; title=${entry#*:}
  marker=data/.recompute_done_$city
  lock=data/.lock_$city
  [ -f $marker ] && return
  mkdir $lock 2>/dev/null || return          # someone else owns this city
  trap "rmdir $lock 2>/dev/null" EXIT
  graph=data/${city}_full.graphml
  dir=data/day_mds_${city}_dir.json
  gold=data/day_mds_${city}_gold.json
  suf=_${city}_gold
  echo "" >>$LOG; echo ">>>>>> $city START ($(date +%H:%M:%S)) <<<<<<" >>$LOG

  [ -f $graph ] || $PY scripts/fetch_city_graph.py $city >>$LOG 2>&1 \
    || { echo "!! $city graph FAIL" >>$LOG; return; }

  if [ -f data/${city}_match_links.json ] && [ -f $dir ]; then
    echo "  $city: reuse existing directions+matching harvest" >>$LOG
  else
    $PY scripts/collect_directions.py $city $graph >>$LOG 2>&1 \
      || { echo "!! $city collect_directions FAIL" >>$LOG; return; }
    $PY scripts/collect_matching.py  $city $graph >>$LOG 2>&1 \
      || echo "!! $city collect_matching FAIL (continuing directions-only)" >>$LOG
    $PY scripts/solve_directions.py  $city $graph >>$LOG 2>&1 \
      || { echo "!! $city solve_directions FAIL" >>$LOG; return; }
  fi

  $PY scripts/solve_sampled.py $dir $gold --anchor 2.5 --shape 0.1 >>$LOG 2>&1 \
    || { echo "!! $city solve_sampled FAIL" >>$LOG; return; }
  $PY scripts/render_base.py $gold $suf >>$LOG 2>&1 \
    || { echo "!! $city render FAIL" >>$LOG; return; }
  $PY scripts/compose_video.py $suf "$title" 1.2 >>$LOG 2>&1 \
    || { echo "!! $city compose FAIL" >>$LOG; return; }
  touch $marker
  echo "  OK -> shots/breathing${suf}.mp4 ($(date +%H:%M:%S))" >>$LOG
}

pids=()
for entry in $CITIES; do
  process_city "$entry" &
  pids+=($!)
  if (( ${#pids} >= MAXJOBS )); then     # block until the oldest finishes -> <=MAXJOBS concurrent
    wait ${pids[1]}
    pids=(${pids[2,-1]})
  fi
done
wait

echo "" >>$LOG; echo "=== full recompute done $(date) ===" >>$LOG
ls -la shots/breathing_*_gold.mp4 >>$LOG 2>&1
