#!/bin/sh
# Download the January-2019 Uber Movement speed matrices (tracebase mirror)
# needed to re-solve the Manhattan and Seattle layouts from scratch.
# Not needed just to re-render the videos - solved layouts are committed.
# ~120 MB total.
set -e
cd "$(dirname "$0")/.."
BASE="https://raw.githubusercontent.com/xinychen/tracebase/main/datasets"

mkdir -p data/seattle
echo "NYC road metadata..."
curl -L -o data/road.csv "$BASE/NYC-movement-data-set/road.csv"
echo "NYC January 2019 speeds (87 MB)..."
curl -L -o data/hourly_speed_mat_2019_1.npz \
    "$BASE/NYC-movement-data-set/hourly_speed_mat_2019_1.npz"
echo "Seattle road metadata..."
curl -L -o data/seattle/road.csv "$BASE/Seattle-movement-data-set/road.csv"
echo "Seattle January 2019 speeds (26 MB)..."
curl -L -o data/seattle/hourly_speed_mat_2019_1.npz \
    "$BASE/Seattle-movement-data-set/hourly_speed_mat_2019_1.npz"
echo "done."
