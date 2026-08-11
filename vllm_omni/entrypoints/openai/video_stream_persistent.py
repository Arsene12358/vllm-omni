# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent streaming-LLM video session (``config.persistent=True``).

Drives ONE long-lived engine streaming request per epoch that ingests frames
as input-only chunks (KV retained across queries, no per-query re-prefill) and
refreshes at the M-RoPE position boundary by re-seeding [sink + recent] into a
new request. With ``engine_rebase`` the engine's streaming-KV rebase keeps
positions bounded instead, so the session is a single epoch-0 request forever
(no driver-side refresh). Complements the windowed re-injection handler in
:mod:`vllm_omni.entrypoints.openai.video_stream_base`, which routes here from
``handle_session`` when the session config sets ``persistent``.
"""

import asyncio
import base64
import json
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from vllm import SamplingParams
from vllm.engine.protocol import StreamingInput
from vllm.logger import init_logger
from vllm.sampling_params import RepetitionDetectionParams, RequestOutputKind

from vllm_omni.entrypoints.openai.video_stream_base import (
    _MAX_FRAME_SIZE,
    _MAX_MSG_QUEUE,
    StreamingVideoSessionConfig,
    _decode_frame_bytes,
)
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.video_stream_base import (
        OmniStreamingVideoHandler,
    )

logger = init_logger(__name__)

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
#
# REFRESH MODE ONLY. v0.26.0 scores repetition_penalty over prompt UNION output
# (vllm/model_executor/layers/utils.py::apply_penalties ->
# apply_repetition_penalties(logits, prompt_mask, output_mask, ...)), and a
# streaming session folds every prior chunk's generated tokens into the prompt
# on each append (vllm/v1/core/sched/scheduler.py::_update_request_as_session:
# prompt_token_ids.extend(kept_output_tokens) then _output_token_ids.clear()).
# Refresh mode re-seeds a fresh request per epoch, so that union stays small and
# 1.3 does what it says — the validated short-epoch value, kept verbatim.
_PERSIST_QUERY_REPETITION_PENALTY = 1.3
# ENGINE-REBASE MODE. One request lives forever, so the prompt union grows
# without bound: within a few answers it covers most of the plausible
# vocabulary, and at temperature 0 a near-uniform 1/1.3 scaling of positive
# logits leaves the argmax ordering untouched — except that it promotes tokens
# the session has NEVER emitted. That is both halves of what V2 measured on
# hardware: no anti-loop effect at all (the "In In In" loop returned despite
# 1.3) plus a standing bias against re-using recall vocabulary a prior answer
# already spent (the reproducible q7 opening-recall confabulation, 15/16 vs
# refresh 16/16 in both parity rounds). So rebase mode drops the penalty and
# guards loops with repetition_detection, which check_stop
# (vllm/v1/core/sched/utils.py) scores over request.output_token_ids — cleared
# on every append, i.e. exactly the answer being generated — and which never
# touches the logits, so verbatim recall is not penalized at all. Firing it
# ends the degenerate answer and, because the session is resumable, leaves the
# session running (vllm/v1/core/sched/scheduler.py::_handle_stopped_request).
# min_count=4 means four exact consecutive repeats of a 1-4 token pattern.
_PERSIST_QUERY_REPETITION_DETECTION = RepetitionDetectionParams(min_pattern_size=1, max_pattern_size=4, min_count=4)
# Stable log needles. Validation harnesses grep these, so treat them as a
# contract: a truncated answer must be distinguishable from a short one, and an
# engine-side session failure from a normal close.
_PERSIST_REPETITION_NEEDLE = "[persistent-session] repetition guard fired"
_PERSIST_ENGINE_ERROR_NEEDLE = "[persistent-session] engine ended the request with an error"
# After a query chunk, ingestion pauses until the answer completes so incoming
# frame chunks can't chop the query's decode (each session update would discard
# the in-flight token, truncating the answer). Resume on the response, or after
# this safety timeout (generation is slow under enforce_eager).
_PERSIST_QUERY_RESPONSE_TIMEOUT = 120.0
# Defense in depth. Once an epoch's input stream closes, the engine owes the
# session a terminal output; without it the client would wait for its own recv
# timeout (600 s in the example client). Bound that tail so a dropped engine
# terminal degrades to a logged warning plus a normal `session.done` instead of
# a hung client. Generous vs. the observed answer decode times (<= 3 s).
_PERSIST_STREAM_END_TIMEOUT = 30.0
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


# --- Qwen prompt seam ---
# The base handler's pipeline hooks (``build_engine_prompt``) render an
# OpenAI-style message list that ``_preprocess_to_engine_prompt`` pushes
# through the chat template, i.e. one *closed* turn per request. A persistent
# session instead streams *partial* turns as raw prompt fragments — a seed that
# opens the user turn, bare video blocks that extend it, and a query that closes
# the user turn and opens the assistant turn — so the hooks do not fit. These
# helpers are the Qwen-specific seam, mirroring how ``serving_video_stream.py``
# keeps Qwen prompt details out of the base handler.


def _qwen_seed_prefix(system_prompt: str) -> str:
    """Chat preamble that opens the persistent session's user turn."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"


def _qwen_query_suffix(text: str) -> str:
    """Close the user turn and open the assistant turn for a query."""
    return f" {text}<|im_end|>\n<|im_start|>assistant\n"


def _decode_frame_to_ndarray(raw_bytes: bytes) -> "np.ndarray":
    """Decode JPEG/PNG bytes to a uint8 [H, W, 3] RGB array (video mm format)."""
    return np.asarray(_decode_frame_bytes(raw_bytes))


class _TextStreamDemux:
    """Stream a persistent session's text output as incremental deltas, robust to the
    omni orchestrator stream:

    - Input-only frame chunks emit empty-text outputs -> ignored.
    - With session pacing a query generates cleanly: the cumulative text grows
      token by token (``finish_reason`` None) until the final token, so we emit a
      ``delta`` per growth and ``done`` at the finish.
    - A finished answer may be re-yielded by the engine -> deduped against the
      text delivered so far.

    The text is cumulative for the LIFE OF THE ENGINE REQUEST, not per answer:
    the output processor's detokenizer is never reset across a session's chunk
    appends (``RequestState.apply_streaming_update`` leaves it alone), so in
    ``engine_rebase`` mode -- one request forever -- the n-th answer arrives as
    ``answer_1 + ... + answer_n``. Refresh mode re-seeds a request per epoch, so
    its answers start from an empty cumulative. Both are handled by baselining
    each answer at whatever was already delivered when it opened: a cumulative
    that still carries the delivered prefix is segmented against it, a fresh one
    (new request) baselines at zero. Without this, every answer re-streamed the
    whole session transcript as deltas and its ``done`` text was the transcript
    (V2 measured done_chars 303 -> 622 -> ... -> 2687, nesting 15/16 in rebase
    mode vs 0/16 in refresh).

    The handler additionally gates these events to one answer per query (so any
    post-answer re-emit is dropped upstream regardless).

    ``feed(cum_text, finish_reason, ntok)`` returns events, each one of
    ``("start",)``, ``("delta", text)``, ``("done", text)``.
    """

    def __init__(self) -> None:
        self._cur = ""  # cumulative text already streamed for the in-flight answer
        self._open = False  # response.start emitted for the in-flight answer
        self._base = ""  # cumulative text the in-flight answer is measured against
        self._delivered = ""  # cumulative text through the last delivered answer

    def feed(self, cum: str, finish_reason: Any, ntok: int) -> list[tuple]:
        cum = cum or ""
        if not cum.strip():
            return []  # input-only / empty output -> ignore
        events: list[tuple] = []
        if not self._open:
            if cum == self._delivered:
                return []  # re-emit of what was already delivered -> ignore
            self._open = True
            self._base = self._delivered if cum.startswith(self._delivered) else ""
            self._cur = self._base
            events.append(("start",))
        delta = cum[len(self._cur) :] if cum.startswith(self._cur) else cum
        if delta:
            events.append(("delta", delta))
        self._cur = cum
        if finish_reason is not None:
            answer = cum[len(self._base) :] if cum.startswith(self._base) else cum
            events.append(("done", answer))
            self._delivered = cum
            self._open = False
            self._base = ""
            self._cur = ""
        return events


def _text_output_state(
    output: OmniRequestOutput,
) -> tuple[str, Any, int, Any] | None:
    """Pull (cumulative_text, finish_reason, n_tokens, stop_reason) from a text output."""
    if getattr(output, "final_output_type", "text") != "text":
        return None
    request_output = getattr(output, "request_output", None)
    if request_output is None:
        return None
    outputs = getattr(request_output, "outputs", None)
    if not isinstance(outputs, list) or not outputs:
        return None
    completion = outputs[0]
    text = getattr(completion, "text", "") or ""
    finish_reason = getattr(completion, "finish_reason", None)
    token_ids = getattr(completion, "token_ids", None)
    ntok = len(token_ids) if token_ids is not None else 0
    # The engine explains a per-request failure here (e.g. the session prompt
    # crossing max_model_len), so carry it through to the client.
    stop_reason = getattr(completion, "stop_reason", None)
    return (text, finish_reason, ntok, stop_reason)


class _ClientGoneError(Exception):
    """The browser went away while the driver was emitting an answer."""


async def _emit_demux_event(websocket: WebSocket, ev: tuple) -> None:
    """Translate a demux event into a WebSocket response message."""
    kind = ev[0]
    if kind == "start":
        await websocket.send_json({"type": "response.start"})
    elif kind == "delta":
        await websocket.send_json({"type": "response.text.delta", "delta": ev[1]})
    elif kind == "done":
        await websocket.send_json({"type": "response.text.done", "text": ev[1]})


async def _close_result_stream(result_gen: Any) -> None:
    """Tear down the engine request behind an epoch's result generator.

    ``AsyncOmni.generate`` aborts its internal requests on ``GeneratorExit``,
    so closing the generator is what releases the engine when the epoch ends
    early (client disconnect, dropped terminal output). A drained generator is
    already closed and this is a no-op.
    """
    aclose = getattr(result_gen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        logger.debug("persistent session: closing the result stream raised", exc_info=True)


async def run_persistent_session(
    handler: "OmniStreamingVideoHandler",
    websocket: WebSocket,
    config: StreamingVideoSessionConfig,
) -> None:
    """Persistent streaming-LLM session with position refresh.

    ONE engine streaming request per epoch ingests video frames as
    input-only chunks (``max_tokens=1``; KV retained by streaming-KV
    eviction, no per-query re-prefill). Queries are generate chunks
    interleaved in receive order. When the open-loop M-RoPE position
    estimate reaches ``refresh_at_position`` (kept below the trained-range
    wall 65536), the epoch ends and a fresh request re-seeds
    [opening + recent] -> position resets to 0, opening recall preserved.
    Validated mechanic: jobs 2051/2069.

    With ``config.engine_rebase`` the engine's ``--streaming-kv-rebase-at``
    keeps M-RoPE positions bounded, so the driver never refreshes: no
    position accounting, no re-seed, ONE epoch-0 engine request for the
    session's lifetime (the input generator returns only on ``done_event``).
    """
    if handler._engine_client is None:
        await handler._send_error(websocket, "Streaming video requires an engine client")
        return

    if "audio" in config.modalities:
        logger.warning("persistent mode is text-output only; ignoring audio output modality")

    sys_prompt = config.system_prompt or _PERSIST_DEFAULT_SYS
    seed_prefix = _qwen_seed_prefix(sys_prompt)
    opening_size = config.sink_frames  # frames pinned as the re-seeded opening
    recent_window = config.num_frames  # frames carried into the re-seed for continuity
    engine_rebase = config.engine_rebase  # engine keeps positions bounded -> never refresh
    if engine_rebase and "refresh_at_position" in config.model_fields_set:
        logger.warning(
            "engine_rebase mode: refresh_at_position=%d is ignored (the engine's "
            "streaming-KV rebase keeps positions bounded; no driver-side refresh)",
            config.refresh_at_position,
        )
    if opening_size == 0:
        logger.warning("persistent mode with sink_frames=0: opening recall not preserved across refresh")

    # Ordered event stream from the WS reader, preserving receive order:
    #   ("frame", ndarray) | ("query", text, max_tokens) | ("done",)
    event_q: asyncio.Queue[tuple] = asyncio.Queue(maxsize=_MAX_MSG_QUEUE)
    done_event = asyncio.Event()
    # Set by the output loop when a query's answer completes; gen() awaits it
    # after a query chunk so frame ingestion can't interrupt the generation.
    query_done_event = asyncio.Event()
    # Armed by gen() when a query is sent; the output loop delivers exactly one
    # answer per query and suppresses any trailing re-emits / degeneration.
    expecting_answer = {"v": False}

    async def _reader() -> None:
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=handler._idle_timeout)
                except asyncio.TimeoutError:
                    await handler._send_error(websocket, "Idle timeout")
                    await event_q.put(("done",))
                    return
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await handler._send_error(websocket, "Invalid JSON")
                    continue
                if not isinstance(msg, dict):
                    await handler._send_error(websocket, "Messages must be JSON objects")
                    continue
                mtype = msg.get("type")
                if mtype == "video.frame":
                    data = msg.get("data", "")
                    if not data:
                        continue
                    if len(data) > _MAX_FRAME_SIZE:
                        await handler._send_error(websocket, "Frame too large")
                        continue
                    try:
                        raw_bytes = base64.b64decode(data, validate=True)
                        arr = await asyncio.to_thread(_decode_frame_to_ndarray, raw_bytes)
                    except Exception:
                        await handler._send_error(websocket, "Invalid image data")
                        continue
                    await event_q.put(("frame", arr))
                elif mtype == "video.query":
                    text = msg.get("text", "")
                    maxt = msg.get("max_tokens")
                    maxt = maxt if isinstance(maxt, int) and maxt > 0 else _PERSIST_QUERY_MAX_TOKENS
                    await event_q.put(("query", text, maxt))
                elif mtype == "video.done":
                    await event_q.put(("done",))
                    return
                elif mtype == "ping":
                    try:
                        await websocket.send_json({"type": "pong"})
                    except Exception:
                        pass
                else:
                    await handler._send_error(websocket, f"Unknown type: {mtype}")
        except WebSocketDisconnect:
            await event_q.put(("done",))
        except Exception:
            await event_q.put(("done",))
            raise

    # 1-item lookahead so a query is never reordered ahead of frames received
    # before it when batching consecutive frames into a chunk.
    stash: list[tuple] = []

    async def _next_event() -> tuple:
        if stash:
            return stash.pop()
        return await event_q.get()

    async def _next_chunk() -> tuple:
        """('frames', [ndarray, ...]) | ('query', text, maxt) | ('done',)."""
        ev = await _next_event()
        if ev[0] != "frame":
            return ev
        batch = [ev[1]]
        while len(batch) < _PERSIST_FRAMES_PER_CHUNK:
            try:
                nxt = event_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if nxt[0] != "frame":
                stash.append(nxt)
                break
            batch.append(nxt[1])
        return ("frames", batch)

    opening: list = []  # first `opening_size` frames (ndarray), captured in epoch 0
    recent: deque = deque(maxlen=recent_window) if recent_window > 0 else deque()
    pos_est = {"v": 0}  # open-loop M-RoPE position estimate (dict for nonlocal mutation)

    def _est(n_frames: int) -> int:
        chunks = max(1, (n_frames + _PERSIST_FRAMES_PER_CHUNK - 1) // _PERSIST_FRAMES_PER_CHUNK)
        return chunks * _PERSIST_EST_POS_PER_CHUNK

    def _chunk(text: str, frames: list | None, max_tokens: int, *, query: bool = False) -> StreamingInput:
        # Penalty scoping is mode-dependent — see the module constants: refresh
        # mode's per-epoch request keeps repetition_penalty meaningful, while
        # rebase mode's forever-request needs an output-scoped guard instead.
        prompt: dict[str, Any] = {"prompt": text}
        if frames:
            prompt["multi_modal_data"] = {"video": np.stack(frames, axis=0)}
        return StreamingInput(
            prompt=prompt,
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                seed=42,
                repetition_penalty=(_PERSIST_QUERY_REPETITION_PENALTY if query and not engine_rebase else 1.0),
                repetition_detection=(_PERSIST_QUERY_REPETITION_DETECTION if query and engine_rebase else None),
                output_kind=RequestOutputKind.CUMULATIVE,
            ),
        )

    default_sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=42,
        output_kind=RequestOutputKind.CUMULATIVE,
    )

    def _ingest_frames(frames: list) -> None:
        # The opening/recent buffers stay maintained in engine_rebase mode too,
        # even though only the refresh re-seed consumes them today: they are
        # bounded (sink_frames / num_frames caps) and keeping ingestion
        # mode-uniform preserves the session state any future teardown or
        # recovery path would need. Position accounting, by contrast, feeds
        # only the refresh check, so engine_rebase skips its bookkeeping.
        for f in frames:
            if len(opening) < opening_size:
                opening.append(f)
            recent.append(f)
        if not engine_rebase:
            pos_est["v"] += _est(len(frames))

    def _epoch_input_stream(epoch: int):
        async def gen():
            # ---- seed ----
            if epoch == 0:
                while True:  # wait for the first real frame chunk to seed
                    first = await _next_chunk()
                    if first[0] == "done":
                        done_event.set()
                        return
                    if first[0] == "query":
                        await handler._send_error(websocket, "No frames buffered")
                        continue
                    break
                _ingest_frames(first[1])
                yield _chunk(seed_prefix + _VID_BLOCK, first[1], 1)
            else:
                # ---- refresh re-seed: opening (sink) then recent (continuity) ----
                pos_est["v"] = 0
                if opening:
                    pos_est["v"] += _est(len(opening))
                    yield _chunk(seed_prefix + _VID_BLOCK, list(opening), 1)
                    if recent:
                        rec = list(recent)
                        pos_est["v"] += _est(len(rec))
                        yield _chunk(_VID_BLOCK, rec, 1)
                elif recent:
                    rec = list(recent)
                    pos_est["v"] += _est(len(rec))
                    yield _chunk(seed_prefix + _VID_BLOCK, rec, 1)
            # ---- ingest loop ----
            # `progressed` gates the refresh on at least one newly-ingested
            # frame chunk this epoch, so a re-seed whose own position cost
            # already meets the threshold can't spin in a zero-progress
            # refresh loop.
            progressed = False
            while True:
                if not engine_rebase and progressed and pos_est["v"] >= config.refresh_at_position:
                    return  # end epoch at a frame boundary -> triggers refresh
                ev = await _next_chunk()
                if ev[0] == "done":
                    done_event.set()
                    return
                if ev[0] == "query":
                    _, text, maxt = ev
                    query_done_event.clear()
                    expecting_answer["v"] = True
                    yield _chunk(_qwen_query_suffix(text), None, maxt, query=True)
                    # Pause ingestion until the answer completes (or times out)
                    # so frame chunks don't truncate the query's decode.
                    try:
                        await asyncio.wait_for(query_done_event.wait(), timeout=_PERSIST_QUERY_RESPONSE_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.warning("persistent query response timed out; resuming ingestion")
                else:
                    _ingest_frames(ev[1])
                    progressed = True
                    yield _chunk(_VID_BLOCK, ev[1], 1)

        return gen()

    async def _tracked_epoch_input_stream(epoch: int, inputs_done: asyncio.Event):
        """Epoch input stream that flags when the engine has drained it."""
        try:
            async for chunk in _epoch_input_stream(epoch):
                yield chunk
        finally:
            inputs_done.set()

    async def _consume_epoch(result_gen, demux: _TextStreamDemux) -> None:
        async for output in result_gen:
            if not isinstance(output, OmniRequestOutput):
                continue
            state = _text_output_state(output)
            if state is None:
                continue
            cum, finish_reason, ntok, stop_reason = state
            if finish_reason == "error":
                # The engine failed this request on its own — a session append
                # that would cross max_model_len is rejected per-request (the
                # scheduler's streaming-overflow guard) rather than taking the
                # EngineCore down. The engine is fine; only this session is
                # over, so report it and let the epoch close cleanly.
                detail = str(stop_reason) if stop_reason else "see the engine log for the reason"
                logger.error("%s: %s", _PERSIST_ENGINE_ERROR_NEEDLE, detail)
                done_event.set()
                query_done_event.set()
                try:
                    await handler._send_error(websocket, f"Persistent session ended by the engine: {detail}")
                except Exception:
                    logger.debug("persistent session: could not send the engine error", exc_info=True)
                return
            if finish_reason == "repetition":
                # The anti-loop guard fired: this answer was cut short at a
                # degenerate repeat, so the transcript is truncated by design and
                # must not read as a model-quality result.
                logger.info("%s: answer truncated after %d tokens", _PERSIST_REPETITION_NEEDLE, ntok)
            for ev in demux.feed(cum, finish_reason, ntok):
                # Deliver exactly one answer per query; suppress trailing
                # re-emits / degeneration once the answer is delivered.
                if not expecting_answer["v"]:
                    continue
                if ev[0] == "done":
                    expecting_answer["v"] = False
                    query_done_event.set()  # unblock ingestion (pacing)
                try:
                    await _emit_demux_event(websocket, ev)
                except Exception as exc:
                    # A failed send means the browser is gone. That ends the
                    # session; it is not an epoch failure, so stop emitting and
                    # unblock the input generator instead of unwinding loudly.
                    done_event.set()
                    query_done_event.set()
                    raise _ClientGoneError from exc

    reader_task = asyncio.create_task(_reader())
    epoch = 0
    try:
        while not done_event.is_set():
            demux = _TextStreamDemux()  # fresh per epoch (new request_id)
            request_id = f"vsess-{uuid.uuid4().hex[:8]}-{epoch}"
            inputs_done = asyncio.Event()
            result_gen = None
            try:
                result_gen = handler._engine_client.generate(
                    prompt=_tracked_epoch_input_stream(epoch, inputs_done),
                    sampling_params=default_sp,
                    request_id=request_id,
                    output_modalities=["text"],
                )
                consume_task = asyncio.create_task(_consume_epoch(result_gen, demux))
                inputs_task = asyncio.create_task(inputs_done.wait())
                try:
                    await asyncio.wait({consume_task, inputs_task}, return_when=asyncio.FIRST_COMPLETED)
                    if consume_task.done():
                        await consume_task
                    else:
                        # The input stream closed, so the engine owes a terminal
                        # output; bound that wait rather than trusting it.
                        await asyncio.wait_for(consume_task, timeout=_PERSIST_STREAM_END_TIMEOUT)
                finally:
                    inputs_task.cancel()
            except _ClientGoneError:
                logger.info("persistent session epoch %d: client disconnected; closing session", epoch)
                break
            except asyncio.TimeoutError:
                logger.warning(
                    "persistent session epoch %d: engine stream did not end within %.0fs of the "
                    "input stream closing (dropped terminal output); closing the session anyway",
                    epoch,
                    _PERSIST_STREAM_END_TIMEOUT,
                )
                break
            except Exception:
                logger.exception("Persistent session epoch %d failed", epoch)
                await handler._send_error(websocket, "Persistent session epoch failed")
                break
            finally:
                # Whatever ended the epoch, release the engine request: a
                # generator left suspended mid-stream would keep it alive.
                await _close_result_stream(result_gen)
            if engine_rebase:
                # ONE engine request per session: the input generator only
                # returns on done_event, so reaching here without it (an
                # engine-side stream end) closes the session instead of
                # re-seeding a second request the mode promises never to make.
                break
            epoch += 1
        try:
            await websocket.send_json({"type": "session.done"})
        except Exception:
            logger.debug("persistent session: could not send session.done (socket closed)")
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
