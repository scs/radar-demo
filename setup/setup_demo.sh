#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
sed "s#FIXME#${SCRIPTPATH}#g" "${SCRIPTPATH}/radar-demo.desktop.template" > "${HOME}/.local/share/applications/radar-demo.desktop"
cp "${SCRIPTPATH}/radar.png" "${HOME}/.local/share/radar.png"
