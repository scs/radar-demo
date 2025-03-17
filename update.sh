#!/bin/bash
git reset --hard
git checkout main
git pull
chmod +x ./update-hook.sh
./.update-hook.sh
