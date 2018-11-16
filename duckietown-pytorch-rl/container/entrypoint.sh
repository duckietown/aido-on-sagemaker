#!/bin/bash

echo "Starting Xvfb"
Xvfb :99 -ac &
sleep 2

echo "Executing command $@"
export DISPLAY=:99

exec "$@"
