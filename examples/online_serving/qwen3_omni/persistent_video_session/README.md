<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM project -->

# Persistent Streaming-Video Session (Qwen3-Omni)

Always-online video understanding: a continuous video stream into Qwen3-Omni, text
answers out, **GPU memory bounded regardless of stream length**, and the stream's
**opening always recallable** across position-refreshes.

This is the `persistent` mode of the `/v1/video/chat/stream` WebSocket handler
(`vllm_omni/entrypoints/openai/serving_video_stream.py`). The default
`persistent: false` keeps the original windowed re-injection behavior; setting
`persistent: true` in the session config drives one engine streaming request that
ingests frames as input-only chunks (KV held flat by streaming-KV eviction, no
per-query re-prefill) and refreshes at the rotary-position boundary by re-seeding
`[opening + recent]`. That driver-side refresh is the default of two position-boundedness modes — with a rebase-capable engine the refresh disappears entirely and one request runs forever (see "Two boundedness modes" below).

Open `assets/showcase.html` for a narrated walkthrough with an embedded ~45 s
recording of a live session (the recording is being re-captured on the v0.26.0
stack — see the validation report until it lands).

![demo](assets/recording.gif)

## What you'll see

| # | Signal | Where |
|---|--------|-------|
| 1 | **Current-moment tracking** — answer `(a)` changes as the video progresses | client output |
| 2 | **Opening retention** — answer `(b)` stays about the opening, across refreshes | client output |
| 3 | **Bounded memory** — KV `alive=N` stays flat while the stream grows | server log |
| 4 | **No crash** — clean `session.done`, refreshes appear as new request ids | client + server log |

Measured on 2×H200 (800-frame session, queries every 50 frames; S2 = eager run,
S3 = CUDA-graphs run of the v0.26.0 validation):

| Metric | old branch (v0.20-era) | this port (v0.26.0) |
|--------|------------------------|---------------------|
| ingestion (session e2e incl. answers) | ~25 frames/s (**~12× a live 2 fps feed**) | **25.5 frames/s** graphs / ~13 frames/s eager |
| query cycle (50-frame ingest + answer) | ~1.5 s query latency | **~1.4 s** graphs / 2.8–4.3 s eager |
| decode | ~35 tok/s eager → ~199 tok/s graphs | 26.5 tok/s eager → **213 tok/s** graphs (per-answer median) |
| KV `alive` | 672–673 | **672–673** (flat across 10 epochs) |
| server ready | — | 182 s eager / **222 s** graphs (torch.compile ~39 s + capture ≤2 s per stage) |

A single server also sustains **concurrent** persistent sessions once
`--max-num-seqs` is raised (validated up to 8 streams; see the validation
report's S4 scaling table).

## Requirements

- **2 GPUs**, enough memory for `Qwen3-Omni-30B-A3B-Instruct` (validated on 2×H200). The
  omni server places stages across the two GPUs — do **not** pass `--tensor-parallel-size`.
- **vLLM v0.26.0 with the streaming-KV overlay.** The eviction (`--streaming-kv-*`) that
  bounds the cache, plus a per-chunk `max_tokens` scheduler fix the persistent mode needs,
  live on [`Arsene12358/vllm@feat/streaming-kv-v026`](https://github.com/Arsene12358/vllm/tree/feat/streaming-kv-v026).
  The changes are pure-Python over the `v0.26.0` release, so the quickest path is to install
  the wheel and overlay the branch's changed files:
  ```bash
  pip install vllm==0.26.0
  git clone -b feat/streaming-kv-v026 https://github.com/Arsene12358/vllm.git vllm-fork
  SITE=$(python -c "import vllm,os; print(os.path.dirname(vllm.__file__))")
  ( cd vllm-fork && git fetch --no-tags https://github.com/vllm-project/vllm.git tag v0.26.0 >/dev/null 2>&1
    for f in $(git diff --name-only v0.26.0..HEAD -- 'vllm/**/*.py'); do
      rel=${f#vllm/}; mkdir -p "$SITE/$(dirname "$rel")"; cp "$f" "$SITE/$rel"; done )
  ```
- **This vLLM-Omni checkout** (the `feat/persistent-video-session-v026` branch, with the
  persistent handler) installed from source:
  ```bash
  git clone -b feat/persistent-video-session-v026 https://github.com/Arsene12358/vllm-omni.git
  cd vllm-omni && pip install -e .
  ```
- Client deps: `pip install websockets pillow` (frame reading uses `vllm.assets.video`,
  or falls back to `opencv-python`).

## Run the demo

```bash
# 1) serve (one terminal)
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct PORT=8901 ./run_server.sh
# wait until http://localhost:8901/v1/models returns a model id
# (first load ~4 min: weights + torch.compile + CUDA-graph capture)

# 2) stream a clip + ask questions (another terminal)
python demo_client.py --video your_clip.mp4 --port 8901 \
    --frames 800 --query-every 50 --refresh-at 2000 --sink 6 --recent 10
# add --brief for tight two-sentence answers
```

Expected: `(a)` answers evolve with the video, `(b)` answers keep describing the same
opening, and the run ends `RESULT: PASS`. Then confirm bounded memory + refreshes in the
server output:

```bash
grep "streaming-kv" server.log | grep -oE "alive=[0-9]+" | sort -u          # tight band, e.g. 672-673
grep -oE "vsess-[a-f0-9]+-[0-9]+" server.log | sed -E 's/.*-([0-9]+)$/\1/' | sort -nu  # 0,1,2,.. = refreshes
```

## Reproduce the recording (`assets/recording.mp4`)

```bash
./run_server.sh > server.log 2>&1 &                 # serve, logging to a file
./monitor.sh server.log >> capture.log &            # sample GPU mem + KV alive @1s
VIDEO_PATH=your_clip.mp4 python capture_client.py >> capture.log   # stream + brief queries
./render_demo.py capture.log --out recording.mp4 --gif             # PIL + ffmpeg dashboard
```

`render_demo.py` draws tokens-processed climbing while the live KV working set and GPU
memory stay flat, with the model's answers and refresh markers. Needs `ffmpeg` + `pillow`.

## Two boundedness modes

KV **memory** is always bounded by the streaming-KV eviction (`--streaming-kv-*`). M-RoPE **positions** are what would otherwise grow with the stream — past the model's trained range (65536) quality collapses — and the session bounds them in one of two ways:

- **Refresh (default — the validated fallback).** The driver estimates positions and, at `refresh_at_position`, ends the engine request and re-seeds a fresh one from `[opening + recent]`. Positions restart every epoch; request ids step `vsess-...-0, -1, -2, ...`. Everything under "Run the demo" uses this mode.
- **Engine rebase (`engine_rebase: true` — unbounded, one request forever).** The engine itself rebases positions in place when they cross `--streaming-kv-rebase-at` (rotating the recent KV window to match), so the driver never refreshes: a single engine request serves the whole session — no epochs, no re-seeding, and the request id stays `vsess-...-0`. If `refresh_at_position` is also set, the driver warns once and ignores it.

Enable it on **both** sides (the server flag alone never triggers — the driver still refreshes first; the client flag alone removes the only position bound):

```bash
# server — add to the serve command (run_server.sh ships this as a commented block;
# the required --max-model-len and --limit-mm-per-prompt raises are explained
# there and below):
#   --streaming-kv-rebase-at 49152 \
# client:
python demo_client.py --video your_clip.mp4 --port 8901 \
    --frames 800 --query-every 50 --engine-rebase --sink 6 --recent 10
# (this example run stays below the rebase threshold — stream longer to see rebases)
# capture_client.py: prefix with ENGINE_REBASE=1
```

**The invariant** (validated at server startup): `rebase_at >= start_size + 2*recent_size`, which keeps consecutive rebases at least one full recent window apart so any cached KV entry is rotated at most once before eviction claims it. The shipped geometry passes with room — `2560 + 2*8192 = 18944 <= 49152` ✓ — and effective positions stay well under the 65536 trained-range wall.

**Raise `--max-model-len` and `--limit-mm-per-prompt` with it.** The request's token count binds ~10.6× before its positions do — measured on the demo clip at ~640px, one video item costs ~23 positions / ~243 tokens, i.e. ~0.095 positions per token (per-item costs are resolution-dependent: ~40 positions/item at ~1280px, which is also the driver's deliberately conservative refresh estimate) — so at the default `--max-model-len 65536` the session is length-capped near position ~6,200 and the first rebase would never fire. Size it to the intended session token horizon, e.g. `--max-model-len 1048576` (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` is already exported by `run_server.sh`; KV memory stays bounded by `--streaming-kv-*`, the added cost is host-side buffers). The video-item limit binds even sooner: reaching the first rebase alone takes `49152/23 ≈ 2150` items (set the limit ≥ ~2200), a 1M-token horizon holds `1048576/243 ≈ 4300` (rule of thumb: `max-model-len/240`, e.g. `'{"video": 5120}'`) — and at ~2 frames per item, **any** rebase-mode run past ~500 frames needs a raise from the shipped `'{"video": 256}'` (refresh mode never hits it: each epoch re-seeds after ~50 items).

**Observability.** Each rebase logs one line in this fixed format (grep-stable, like the eviction line):

```
[streaming-kv] rebase req=<id> delta=<int> new_base=<int> recent_tokens=<int>
```

```bash
grep "streaming-kv. rebase" server.log                    # one line per rebase, once positions cross 49152
grep -oE "vsess-[a-f0-9]+-[0-9]+" server.log | sort -u    # stays vsess-...-0: no refresh re-seeds
```

Engine-rebase mode needs the rebase overlay branches: [`Arsene12358/vllm@feat/streaming-kv-rebase-v026`](https://github.com/Arsene12358/vllm/tree/feat/streaming-kv-rebase-v026) (a superset of `feat/streaming-kv-v026`; same pure-Python overlay recipe as in "Requirements", substituting the branch name) and this vLLM-Omni branch (`feat/persistent-rebase-v026`). Refresh mode runs on the base branches unchanged.

## Config / tuning knobs

| Knob | Where | Effect |
|------|-------|--------|
| `sink_frames` (`--sink`) | session config | opening frames pinned + re-seeded each epoch (the retained opening) |
| `num_frames` (`--recent`) | session config | recent frames re-seeded at a refresh for continuity |
| `refresh_at_position` (`--refresh-at`, ≥1024) | session config | larger → longer epochs / fewer refreshes (keep epoch tokens < `max-model-len`) |
| `engine_rebase: true` (`--engine-rebase` / `ENGINE_REBASE=1`) | session config | trust the engine's position rebase: one request forever, no driver refreshes; `refresh_at_position` ignored (see "Two boundedness modes") |
| `--streaming-kv-start-size` / `-recent-size` | server | KV tokens pinned (opening) / kept (recent) — the memory bound |
| `--streaming-kv-rebase-at` | server | engine-side position-rebase threshold (`engine_rebase` mode); must be ≥ `start + 2*recent` |
| `--max-num-seqs` | server | concurrent persistent sessions per server; the demo default `1` keeps single-viewer latency, `8` validated with 8 concurrent streams on 2×H200 (each stream keeps its own flat 672–673 KV band) |
| `system_prompt` (`--brief`) | session config | brevity instruction for clean two-sentence answers |
| `persistent: false` | session config | the original windowed re-injection handler (unchanged) |

One epoch accumulates about `refresh_at_position / 40` video items in a single request (40 is the driver's per-chunk M-RoPE position estimate), so keep `refresh_at_position / 40` at or below the server's `--limit-mm-per-prompt` video limit. The shipped pair is safe: `--refresh-at 2000` → ~50 items against `--limit-mm-per-prompt '{"video": 256}'`. Raise the server limit before raising `--refresh-at`.

## Troubleshooting

- **Streaming session wedges mid-stream under `--enforce-eager`** — eager + the default
  async scheduling is a broken combination for persistent streaming sessions. If you must
  run eager, set `async_scheduling: false` on stages 0 and 1 **through a deploy config**
  (`--deploy-config your.yaml`): the deploy value picks the sync scheduler class and the
  engine arg together. The `--no-async-scheduling` CLI flag is **not** sufficient — it
  flips only the engine arg while the stage keeps the async scheduler class resolved from
  the deploy YAML (an untested mismatch). The default CUDA-graphs config (this
  `run_server.sh`) needs no override.
- **Server aborts with `local_world_size (2) > visible devices (1)`** — you passed
  `--tensor-parallel-size`; remove it (the omni server self-places stages).
- **Queries return one word / stop immediately** — the vLLM-core `max_tokens` scheduler
  fix isn't applied; re-check the streaming-KV overlay recipe above.
- **Multi-modal limit error mid-stream** — raise `--limit-mm-per-prompt '{"video": N}'`.
- **Verbose "Detailed Breakdown" tail** — base-model behavior; use `--brief`. Not a mechanic issue.

## Files

| File | Purpose |
|------|---------|
| `run_server.sh` | serve Qwen3-Omni with the persistent-mode flags |
| `demo_client.py` | interactive demo: stream a clip, ask questions, verify the 4 signals |
| `capture_client.py` | recording: stream + brief queries, timestamped answer events |
| `monitor.sh` | recording: sample GPU-0 memory + live KV blocks @1s |
| `render_demo.py` | recording: render the captured timeline to MP4/GIF |
| `assets/showcase.html` | narrated walkthrough with the embedded recording (re-captured on the v0.26.0 stack) |
| `assets/recording.mp4` / `.gif` | the captured ~45 s segment (re-captured on the v0.26.0 stack) |
