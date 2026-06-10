#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Sample GPU-0 memory + the live KV working set once a second, emitting MON lines that
# render_demo.py aligns with the capture client's answer events to build the recording.
#
# Run it while the server (with its stdout redirected to SERVER_LOG) and capture_client
# are active, e.g.:
#   ./run_server.sh > server.log 2>&1 &
#   ./monitor.sh server.log >> capture.log &
#   VIDEO_PATH=clip.mp4 python capture_client.py >> capture.log
#   ./render_demo.py capture.log --out recording.mp4 --gif
set +e   # a missing eviction line early on must not stop the loop

SERVER_LOG="${1:-server.log}"
while true; do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  EVICT=$(grep -a 'streaming-kv. eviction' "$SERVER_LOG" 2>/dev/null | tail -1 \
          | grep -oE 'computed=[0-9]+ total_blocks=[0-9]+ alive=[0-9]+' || true)
  echo "MON t=$(date +%s.%3N) mem=$MEM $EVICT"
  sleep 1
done
