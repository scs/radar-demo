#!/bin/bash
git reset --hard
git checkout main
git pull --all
chmod +x .update-hook.sh
./.update-hook.sh
