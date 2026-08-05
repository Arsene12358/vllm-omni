# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent streaming-LLM video session (``config.persistent=True``).

Building blocks for driving ONE long-lived engine streaming request that
ingests frames as input-only chunks (KV retained across queries, no per-query
re-prefill) and refreshes at the M-RoPE position boundary by re-seeding
[sink + recent] into a new request. Complements the windowed re-injection
handler in :mod:`vllm_omni.entrypoints.openai.video_stream_base`.
"""

from typing import Any

# --- persistent streaming-LLM session (config.persistent=True) ---
# NOTE: the values and workarounds below were measured/validated on the old
# harness (job numbers cited in the comments). Each carries a re-verification
# TODO against v0.26.0; these are resolved in validation stage S2
# (omni single-stream).
_PERSIST_FRAMES_PER_CHUNK = 2  # frames per input-only chunk (matches validated harness)
_PERSIST_QUERY_MAX_TOKENS = 300  # default generation length for a query chunk
# Greedy decoding in the long persistent-session context can fall into a
# token-repetition loop after the answer ("... In In In"); a mild repetition
# penalty makes the model emit EOS cleanly instead.
_PERSIST_QUERY_REPETITION_PENALTY = 1.3
# After a query chunk, ingestion pauses until the answer completes so incoming
# frame chunks can't chop the query's decode (each session update would discard
# the in-flight token, truncating the answer). Resume on the response, or after
# this safety timeout (generation is slow under enforce_eager).
_PERSIST_QUERY_RESPONSE_TIMEOUT = 120.0
# Open-loop M-RoPE position estimate. ~40 positions per 2-frame chunk at ~640px /
# 2fps (measured: position ~= 0.056 x tokens, job 2051). Spatial tax dominates per
# chunk, so we estimate per chunk rather than per token; conservative so refresh
# fires before the trained-range wall (65536).
_PERSIST_EST_POS_PER_CHUNK = 40
_PERSIST_DEFAULT_SYS = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs."
)
# Video block with a trailing newline — the Qwen3-Omni get_mrope_input_positions
# off-by-one workaround (verified single+multi-block, jobs 2050/2051/2069).
_VID_BLOCK = "<|vision_start|><|video_pad|><|vision_end|>\n"


class _TextStreamDemux:
    """Stream a persistent session's text output as incremental deltas, robust to the
    omni orchestrator stream:

    - Input-only frame chunks emit empty-text outputs -> ignored.
    - With session pacing a query generates cleanly: the cumulative text grows
      token by token (``finish_reason`` None) until the final token, so we emit a
      ``delta`` per growth and ``done`` at the finish.
    - A finished answer may be re-yielded by the engine -> deduped against the last
      delivered answer.

    The handler additionally gates these events to one answer per query (so any
    post-answer re-emit is dropped upstream regardless).

    ``feed(cum_text, finish_reason, ntok)`` returns events, each one of
    ``("start",)``, ``("delta", text)``, ``("done", text)``.
    """

    def __init__(self) -> None:
        self._cur = ""  # cumulative text already streamed for the in-flight answer
        self._open = False  # response.start emitted for the in-flight answer
        self._last_answer: str | None = None  # last fully-delivered answer (dedup re-emits)

    def feed(self, cum: str, finish_reason: Any, ntok: int) -> list[tuple]:
        cum = cum or ""
        if not cum.strip():
            return []  # input-only / empty output -> ignore
        events: list[tuple] = []
        if not self._open:
            if cum == self._last_answer:
                return []  # re-emit of the delivered answer -> ignore
            self._open = True
            self._cur = ""
            events.append(("start",))
        delta = cum[len(self._cur) :] if cum.startswith(self._cur) else cum
        if delta:
            events.append(("delta", delta))
        self._cur = cum
        if finish_reason is not None:
            events.append(("done", cum))
            self._last_answer = cum
            self._open = False
            self._cur = ""
        return events
