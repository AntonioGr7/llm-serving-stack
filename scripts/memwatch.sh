#!/usr/bin/env bash
# Sample host RAM used (MiB) every 5 s: scripts/memwatch.sh > logs/mem-<label>.log
while true; do echo "$(date +%s) $(free -m | awk '/^Mem:/{print $3}')"; sleep 5; done
