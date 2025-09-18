#!/usr/bin/env bash
set -euo pipefail

BASEDIR="../dataset/zurich_city_02_a"

# parallel arrays of paths and URLs
relpaths=(
  image/left
  image/timestamps.txt
  image/exposure_timestamps_left.txt
  event/left
  event/right
  flow/forward
  flow/backward
  flow/forward_timestamps.txt
  flow/backward_timestamps.txt
  calibration
)

urls=(
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_images_rectified_left.zip
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_image_timestamps.txt
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_image_exposure_timestamps_left.txt
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_events_left.zip
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_events_right.zip
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_optical_flow_forward_event.zip
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_optical_flow_backward_event.zip
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_optical_flow_forward_timestamps.txt
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_optical_flow_backward_timestamps.txt
  https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_02_a/zurich_city_02_a_calibration.zip
)

for i in "${!relpaths[@]}"; do
  relpath=${relpaths[i]}
  url=${urls[i]}
  dest="$BASEDIR/$relpath"

  if [[ $url == *.zip ]]; then
    mkdir -p "$dest"
    zipfile="$dest/$(basename "$url")"
    echo "Downloading ZIP → $zipfile"
    wget -c "$url" -O "$zipfile"
    echo "Extracting → $dest"
    unzip -o "$zipfile" -d "$dest"
    echo "Removing ZIP → $zipfile"
    rm -f "$zipfile"
  else
    mkdir -p "$(dirname "$dest")"
    echo "Downloading TXT → $dest"
    wget -c "$url" -O "$dest"
  fi
done

echo "All done!"