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
#
# OPTIONAL — engine-rebase mode (unbounded session; the default launch below stays
# refresh mode). With the flag below the ENGINE keeps M-RoPE positions bounded by
# rebasing them in place, so one request runs for the whole session; enable it by
# adding the flag to the command below AND passing --engine-rebase to the client
# (see README "Two boundedness modes"):
#
#   --streaming-kv-rebase-at 49152 \
#
# 49152 satisfies the single-rotation invariant for the shipped geometry —
# rebase_at >= start + 2*recent: 2560 + 2*8192 = 18944 <= 49152 ✓ — and keeps
# effective positions well under the 65536 trained-range wall.
# ALSO raise --max-model-len: the request's token count binds ~10.6x before its
# positions do — measured on the demo clip at ~640px, one video item costs
# ~23 positions / ~243 tokens, i.e. ~0.095 positions/token (per-item costs are
# resolution-dependent; ~40 positions/item at ~1280px) — so at the default 65536
# the session is length-capped near position ~6,200, before the first rebase would
# ever fire. Size it to the intended session token horizon, e.g.
# --max-model-len 1048576 (VLLM_ALLOW_LONG_MAX_MODEL_LEN is already exported above;
# KV memory stays bounded by --streaming-kv-*, the extra cost is host-side buffers).
# AND raise --limit-mm-per-prompt: reaching the first rebase alone takes
# 49152/23 ~= 2150 video items (set it >= ~2200), a 1M-token horizon holds
# 1048576/243 ~= 4300 (rule of thumb: max-model-len/240, e.g. '{"video": 5120}')
# — and at ~2 frames/item, ANY rebase-mode run past ~500 frames needs a raise
# from the shipped '{"video": 256}'.
vllm serve "$MODEL" --omni --port "$PORT" \
  --trust-remote-code --max-num-seqs 1 \
  --max-model-len 65536 \
  --streaming-kv-start-size 2560 --streaming-kv-recent-size 8192 \
  --limit-mm-per-prompt '{"video": 256}'
