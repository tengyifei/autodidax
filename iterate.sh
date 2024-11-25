#!/bin/bash

set -e

MONITOR_DIR="/$(realpath -m --relative-to / .)"

echo "Monitoring $MONITOR_DIR"

pushd "$MONITOR_DIR"

# Check if inotifywait is installed
if ! command -v inotifywait &> /dev/null; then
  echo "inotifywait not found. Installing inotify-tools..."
  sudo apt install -y inotify-tools
fi

echo "Running test"
pytest "$@" || :
echo "=========== DONE ==========="

while inotifywait -r -e modify,create "$MONITOR_DIR" ; do
  echo "Running test"
  pytest "$@" || :
  echo "=========== DONE ==========="
done

popd
