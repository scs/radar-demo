#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd "${SCRIPTPATH}"/backend || exit 1
eval "$(direnv export bash)"
killall python3
direnv allow .
poetry update
python3 run.py &
sleep 10
chromium-browser --app=http://127.0.0.1:5000 --start-fullscreen
