#!/bin/bash
git reset --hard
git checkout main
git pull --all
git lfs install
git lfs pull
chmod +x .update-hook.sh
./.update-hook.sh
