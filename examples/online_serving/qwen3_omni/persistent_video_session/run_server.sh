#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Serve Qwen3-Omni for the persistent streaming-video session.
#
# Requires vLLM v0.26.0 with the streaming-KV overlay (see README.md) — it
# adds the KV eviction (`--streaming-kv-*`) that bounds the cache and the per-chunk
# `max_tokens` scheduler fix the persistent mode relies on.
#
# Usage:   MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct PORT=8901 ./run_server.sh
# Logs to stdout; redirect to a file if you also run monitor.sh (for the recording):
#          ./run_server.sh > server.log 2>&1 &
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${PORT:-8901}"

# max-model-len stays at the model-native 65536; one persistent epoch fits well under it.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# NOTE: do NOT pass --tensor-parallel-size. The omni server places each stage on its
# own GPU (thinker -> cuda:0, talker/code2wav -> cuda:1); forcing TP makes a stage
# demand 2 visible devices when it only sees 1 and the server aborts at startup.
#
# NOTE: CUDA graphs (the default — no --enforce-eager) is the validated production
# config. If you must run eager: eager + the default async scheduling wedges
# persistent streaming sessions. Set `async_scheduling: false` on stages 0/1 through
# a deploy config (--deploy-config <yaml>) — the --no-async-scheduling CLI flag is
# NOT sufficient (it flips the engine arg but leaves the async scheduler class
# resolved from the deploy YAML). See README "Troubleshooting".
vllm serve "$MODEL" --omni --port "$PORT" \
  --trust-remote-code --max-num-seqs 1 \
  --max-model-len 65536 \
  --streaming-kv-start-size 2560 --streaming-kv-recent-size 8192 \
  --limit-mm-per-prompt '{"video": 256}'
