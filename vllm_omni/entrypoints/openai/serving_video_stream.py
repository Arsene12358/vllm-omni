# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WebSocket handler for streaming video input understanding.

Accepts video frames incrementally via WebSocket, buffers them, and
generates text + optional audio responses using the existing Qwen3-Omni
multi-stage pipeline (thinker -> talker -> code2wav).

Protocol:
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "..."}  # base64 JPEG/PNG frame
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.audio.delta", "data": "...", "format": "wav"}
        {"type": "response.audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}
"""

import asyncio
import base64
import hashlib
import io
import json
import time as _time
import uuid
import wave
from collections import deque
from typing import Any

import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from vllm import SamplingParams
from vllm.engine.protocol import StreamingInput
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

from vllm_omni.entrypoints.openai import video_stream_envs
from vllm_omni.entrypoints.openai.video_frame_filter import FrameSimilarityFilter
from vllm_omni.entrypoints.openai.video_stream_context import (
    text_only_message,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10MB per frame
_MAX_BUFFER_FRAMES = 64
_MAX_AUDIO_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_MSG_QUEUE = 200
_CODEC_FRAME_SAMPLES = 1920  # CausalConv leading-edge artifact length
_BAD_FRAME = object()

# --- persistent streaming-LLM session (config.persistent=True) ---
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


def _decode_frame_bytes(raw_bytes: bytes) -> Any:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _decode_frame_to_ndarray(raw_bytes: bytes) -> "np.ndarray":
    """Decode JPEG/PNG bytes to a uint8 [H, W, 3] RGB array (video mm format)."""
    return np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))


class StreamingVideoSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message."""

    model: str | None = None
    modalities: list[str] = Field(
        default_factory=lambda: ["text", "audio"],
        description="Output modalities: 'text', 'audio', or both.",
    )
    num_frames: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Max frames to sample from buffer for the model.",
    )
    max_frames: int = Field(
        default=50,
        ge=1,
        le=256,
        description="Max frames to keep in the buffer.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt.",
    )
    use_audio_in_video: bool = Field(
        default=True,
        description="Interleave audio chunks with video frames when audio input is present.",
    )
    sampling_params_list: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-stage sampling params [thinker, talker, code2wav].",
    )
    enable_frame_filter: bool = Field(
        default=True,
        description="EVS pixel-similarity pre-filter to drop near-duplicate frames.",
    )
    frame_filter_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="EVS similarity threshold (higher = keep more frames).",
    )
    sink_frames: int = Field(
        default=0,
        ge=0,
        le=64,
        description="StreamingLLM-style sink: pin the first N retained frames "
        "(the stream's opening) and re-inject them into every query so the "
        "model can always reference the opening even after the buffer churns "
        "past max_frames. 0 (default) = current windowed behavior.",
    )
    persistent: bool = Field(
        default=False,
        description="Persistent streaming-LLM session: instead of a fresh "
        "re-injection request per query, drive ONE engine streaming request "
        "that ingests frames as input-only chunks (KV retained, no per-query "
        "re-prefill) and refreshes at the M-RoPE position boundary by "
        "re-seeding [sink+recent] into a new request. False (default) = "
        "windowed re-injection (the shipped fallback).",
    )
    refresh_at_position: int = Field(
        default=60000,
        ge=1024,
        le=262144,
        description="Persistent mode only: estimated M-RoPE position at which "
        "to refresh (re-seed). Kept below the trained-range wall (65536). The "
        "handler estimates position from ingested chunks (~spatial-tax/chunk).",
    )


class _TextStreamDemux:
    """Demux a persistent streaming session's text output into per-query responses.

    Per-output classification, robust to the omni orchestrator stream:
    - Input-only frame chunks (``max_tokens=1``) emit a 1-token throwaway. Ignored
      entirely (``ntok <= 1``) so they never disturb query tracking.
    - After a query finishes, the omni engine re-yields that finished answer once
      per subsequent input chunk (the per-request detokenizer text is unchanged).
      Dedup those by full-answer equality against the last delivered answer — NOT
      by a rolling token, since the interleaved throwaways would defeat that.
    - A query answer is a finished (``finish_reason is not None``), multi-token,
      non-empty generation whose text differs from the last delivered answer.

    Emitted at finish granularity (``start`` + one ``delta`` + ``done``); the user
    deferred incremental token streaming to the realtime-perf milestone.

    ``feed(cum_text, finish_reason, ntok)`` returns events, each one of
    ``("start",)``, ``("delta", text)``, ``("done", text)``.
    """

    def __init__(self) -> None:
        self._last_answer: str | None = None  # last fully-delivered query answer (dedup)

    def feed(self, cum: str, finish_reason: Any, ntok: int) -> list[tuple]:
        cum = cum or ""
        # Only a finished, real (multi-token), non-empty generation that differs
        # from the last delivered answer is a query response; everything else
        # (throwaways, re-emits, in-progress outputs) is ignored.
        if finish_reason is None or ntok <= 1 or not cum.strip():
            return []
        if cum == self._last_answer:
            return []
        self._last_answer = cum
        return [("start",), ("delta", cum), ("done", cum)]


class OmniStreamingVideoHandler:
    """Handles WebSocket sessions for streaming video input.

    Supports:
    - Concurrent frame reception during query processing (reader/processor split)
    - PCM audio input (``audio.chunk``)
    - Async-chunk incremental audio output via ``engine_client.generate()``
    - Multi-turn conversation history
    - Soft interrupt (new query cancels current generation)
    """

    def __init__(
        self,
        chat_service: Any,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
        engine_client: Any | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout
        self._engine_client = engine_client

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return

            if config.persistent:
                await self._run_persistent_session(websocket, config)
                return

            frame_buffer: list[str] = []  # base64-encoded JPEG frames
            # B' sink: the first `sink_frames` retained frames (the stream's
            # opening), pinned and re-injected into every query. Never evicted.
            sink_buffer: list[str] = []
            # Per-frame PIL cache + uuid for mm_hash reuse. Aligned with frame_buffer by index.
            frame_pil_cache: dict[str, tuple[Any, str] | object] = {}  # b64 -> (PIL.Image, uuid) or _BAD_FRAME
            frame_filter = (
                FrameSimilarityFilter(threshold=config.frame_filter_threshold) if config.enable_frame_filter else None
            )
            audio_buffer = bytearray()  # raw PCM16 16kHz mono
            message_history: list[dict[str, Any]] = []
            active_request_id: str | None = None
            prev_request_id: str | None = None  # abort target iff prev was interrupted
            prev_was_interrupted: bool = False
            interrupt_event = asyncio.Event()
            prewarm_tasks: set[asyncio.Task[Any]] = set()
            query_task: asyncio.Task[Any] | None = None

            msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=_MAX_MSG_QUEUE)

            async def _reader() -> None:
                """Receive WebSocket messages and enqueue them."""
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                websocket.receive_text(),
                                timeout=self._idle_timeout,
                            )
                        except asyncio.TimeoutError:
                            await self._send_error(websocket, "Idle timeout")
                            await msg_queue.put(None)
                            return

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            await self._send_error(websocket, "Invalid JSON")
                            continue

                        if not isinstance(msg, dict):
                            await self._send_error(websocket, "Messages must be JSON objects")
                            continue

                        msg_type = str(msg.get("type", ""))
                        if msg_type.startswith("_internal."):
                            await self._send_error(websocket, f"Unknown type: {msg_type}")
                            continue

                        await msg_queue.put(msg)
                        if msg.get("type") == "video.done":
                            return
                except WebSocketDisconnect:
                    await msg_queue.put(None)
                except Exception:
                    await msg_queue.put(None)
                    raise

            async def _cancel_active_query(*, abort_now: bool = False) -> None:
                """Signal soft interrupt for the active query."""
                nonlocal active_request_id, prev_was_interrupted, query_task
                if active_request_id is not None:
                    interrupt_event.set()
                    prev_was_interrupted = True
                    logger.info("Interrupt signaled for %s", active_request_id)
                    if abort_now and self._engine_client:
                        try:
                            await self._engine_client.abort(active_request_id)
                        except Exception:
                            logger.debug("Abort failed for %s", active_request_id, exc_info=True)
                    if query_task is not None and not query_task.done():
                        query_task.cancel()
                        await asyncio.gather(query_task, return_exceptions=True)
                    query_task = None

            async def _processor() -> None:
                """Process enqueued messages."""
                nonlocal active_request_id, prev_request_id, prev_was_interrupted, query_task

                while True:
                    msg = await msg_queue.get()
                    if msg is None:
                        await _cancel_active_query(abort_now=True)
                        return

                    msg_type = msg.get("type")

                    if msg_type == "_internal.frame_decode_failed":
                        frame_data = msg.get("b64", "")
                        removed = frame_data in frame_buffer
                        if removed:
                            frame_buffer[:] = [f for f in frame_buffer if f != frame_data]
                        if frame_pil_cache.get(frame_data) is _BAD_FRAME:
                            frame_pil_cache.pop(frame_data, None)
                        if removed:
                            await self._send_error(websocket, "Frame decode failed")

                    elif msg_type == "video.frame":
                        frame_data = msg.get("data", "")
                        if not frame_data:
                            continue
                        if len(frame_data) > _MAX_FRAME_SIZE:
                            await self._send_error(websocket, "Frame too large")
                            continue
                        try:
                            raw_bytes = base64.b64decode(frame_data, validate=True)
                        except Exception:
                            await self._send_error(websocket, "Invalid image data")
                            continue
                        if frame_filter is not None:
                            try:
                                if not frame_filter.should_retain(raw_bytes):
                                    continue
                            except Exception:
                                await self._send_error(websocket, "Invalid image data")
                                continue
                        max_buf = config.max_frames
                        if len(frame_buffer) >= max_buf:
                            dropped = frame_buffer.pop(0)
                            # Keep the PIL/uuid cache entry alive if the dropped
                            # frame is still pinned as a B' sink frame.
                            if dropped not in sink_buffer:
                                frame_pil_cache.pop(dropped, None)
                        frame_buffer.append(frame_data)
                        # B' sink: pin the first `sink_frames` retained frames.
                        if len(sink_buffer) < config.sink_frames:
                            sink_buffer.append(frame_data)
                        # Prewarm: decode PIL off the event loop so query-time chat_template
                        # can skip base64+Image.open. uuid=md5 lets mm_cache dedupe identical frames.
                        if frame_data not in frame_pil_cache:
                            mm_uuid = hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest()

                            async def _prewarm(b64: str, b: bytes, u: str) -> None:
                                try:
                                    pil = await asyncio.to_thread(_decode_frame_bytes, b)
                                    frame_pil_cache[b64] = (pil, u)
                                except Exception:
                                    frame_pil_cache[b64] = _BAD_FRAME
                                    logger.warning("prewarm decode failed for frame (len=%d)", len(b))
                                    try:
                                        msg_queue.put_nowait({"type": "_internal.frame_decode_failed", "b64": b64})
                                    except asyncio.QueueFull:
                                        logger.warning(
                                            "frame decode failure event dropped because message queue is full"
                                        )

                            task = asyncio.create_task(_prewarm(frame_data, raw_bytes, mm_uuid))
                            prewarm_tasks.add(task)
                            task.add_done_callback(prewarm_tasks.discard)

                    elif msg_type == "audio.chunk":
                        data_b64 = msg.get("data", "")
                        try:
                            pcm_bytes = base64.b64decode(data_b64)
                        except Exception:
                            continue
                        if len(audio_buffer) + len(pcm_bytes) > _MAX_AUDIO_BUFFER_BYTES:
                            await self._send_error(websocket, "Audio buffer overflow")
                            audio_buffer.clear()
                            continue
                        audio_buffer.extend(pcm_bytes)

                    elif msg_type == "video.query":
                        await _cancel_active_query()

                        query_text = msg.get("text", "")
                        audio_data_b64 = msg.get("audio_data")
                        if audio_data_b64:
                            try:
                                decoded = base64.b64decode(audio_data_b64)
                                if len(audio_buffer) + len(decoded) <= _MAX_AUDIO_BUFFER_BYTES:
                                    audio_buffer.extend(decoded)
                                else:
                                    await self._send_error(websocket, "Audio buffer overflow")
                                    audio_buffer.clear()
                            except Exception:
                                pass

                        if not frame_buffer:
                            await self._send_error(websocket, "No frames buffered")
                            continue

                        # Abort only if the previous turn was interrupted mid-flight.
                        # A naturally-finished request is already released by the scheduler;
                        # aborting it again can race with stage-1/2 tear-down and has been
                        # observed to crash flash_attn with a mixed prefill+decode batch
                        # (scheduler_metadata shape mismatch) under longer sessions.
                        if prev_was_interrupted and prev_request_id and self._engine_client:
                            try:
                                await self._engine_client.abort(prev_request_id)
                            except Exception:
                                pass
                            await asyncio.sleep(0.1)
                        prev_was_interrupted = False

                        request_id = f"video-{uuid.uuid4().hex[:12]}"
                        active_request_id = request_id
                        interrupt_event.clear()
                        query_frames = list(frame_buffer)
                        query_sink = list(sink_buffer)  # B' pinned opening frames
                        query_audio_buffer = bytearray(audio_buffer)
                        audio_buffer.clear()
                        query_prewarmed_frames = dict(frame_pil_cache)

                        async def _run_query() -> None:
                            nonlocal active_request_id, prev_request_id
                            try:
                                await self._process_query(
                                    websocket,
                                    config,
                                    query_frames,
                                    query_sink,
                                    query_audio_buffer,
                                    message_history,
                                    query_text,
                                    request_id,
                                    interrupt_event,
                                    query_prewarmed_frames,
                                )
                            finally:
                                if active_request_id == request_id:
                                    prev_request_id = request_id
                                    active_request_id = None

                        query_task = asyncio.create_task(_run_query())

                    elif msg_type == "video.done":
                        if query_task is not None and not query_task.done():
                            await asyncio.gather(query_task, return_exceptions=True)
                            query_task = None
                        await websocket.send_json({"type": "session.done"})
                        return

                    elif msg_type == "ping":
                        try:
                            await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass

                    else:
                        await self._send_error(websocket, f"Unknown type: {msg_type}")

            reader_task = asyncio.create_task(_reader())
            try:
                await _processor()
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
                for t in list(prewarm_tasks):
                    t.cancel()
                if prewarm_tasks:
                    await asyncio.gather(*prewarm_tasks, return_exceptions=True)
                if query_task is not None and not query_task.done():
                    await _cancel_active_query(abort_now=True)

        except WebSocketDisconnect:
            logger.info("Streaming video: client disconnected")
        except Exception as e:
            logger.exception("Streaming video session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Persistent streaming-LLM session (config.persistent=True)
    # ------------------------------------------------------------------

    async def _run_persistent_session(
        self,
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
        """
        if self._engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

        if "audio" in config.modalities:
            logger.warning("persistent mode is text-output only; ignoring audio output modality")

        sys_prompt = config.system_prompt or _PERSIST_DEFAULT_SYS
        seed_prefix = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n"
        opening_size = config.sink_frames  # frames pinned as the re-seeded opening
        recent_window = config.num_frames  # frames carried into the re-seed for continuity
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
                        raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._idle_timeout)
                    except asyncio.TimeoutError:
                        await self._send_error(websocket, "Idle timeout")
                        await event_q.put(("done",))
                        return
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        await self._send_error(websocket, "Invalid JSON")
                        continue
                    if not isinstance(msg, dict):
                        await self._send_error(websocket, "Messages must be JSON objects")
                        continue
                    mtype = msg.get("type")
                    if mtype == "video.frame":
                        data = msg.get("data", "")
                        if not data:
                            continue
                        if len(data) > _MAX_FRAME_SIZE:
                            await self._send_error(websocket, "Frame too large")
                            continue
                        try:
                            raw_bytes = base64.b64decode(data, validate=True)
                            arr = await asyncio.to_thread(_decode_frame_to_ndarray, raw_bytes)
                        except Exception:
                            await self._send_error(websocket, "Invalid image data")
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
                        await self._send_error(websocket, f"Unknown type: {mtype}")
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

        def _chunk(text: str, frames: list | None, max_tokens: int, rep_penalty: float = 1.0) -> StreamingInput:
            prompt: dict[str, Any] = {"prompt": text}
            if frames:
                prompt["multi_modal_data"] = {"video": np.stack(frames, axis=0)}
            return StreamingInput(
                prompt=prompt,
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    seed=42,
                    repetition_penalty=rep_penalty,
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
            for f in frames:
                if len(opening) < opening_size:
                    opening.append(f)
                recent.append(f)
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
                            await self._send_error(websocket, "No frames buffered")
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
                    if progressed and pos_est["v"] >= config.refresh_at_position:
                        return  # end epoch at a frame boundary -> triggers refresh
                    ev = await _next_chunk()
                    if ev[0] == "done":
                        done_event.set()
                        return
                    if ev[0] == "query":
                        _, text, maxt = ev
                        query_done_event.clear()
                        expecting_answer["v"] = True
                        yield _chunk(
                            f" {text}<|im_end|>\n<|im_start|>assistant\n",
                            None,
                            maxt,
                            _PERSIST_QUERY_REPETITION_PENALTY,
                        )
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

        reader_task = asyncio.create_task(_reader())
        epoch = 0
        try:
            while not done_event.is_set():
                demux = _TextStreamDemux()  # fresh per epoch (new request_id)
                request_id = f"vsess-{uuid.uuid4().hex[:8]}-{epoch}"
                try:
                    result_gen = self._engine_client.generate(
                        prompt=_epoch_input_stream(epoch),
                        sampling_params=default_sp,
                        request_id=request_id,
                        output_modalities=["text"],
                    )
                    async for output in result_gen:
                        if not isinstance(output, OmniRequestOutput):
                            continue
                        state = self._text_output_state(output)
                        if state is None:
                            continue
                        cum, finish_reason, ntok = state
                        for ev in demux.feed(cum, finish_reason, ntok):
                            # Deliver exactly one answer per query; suppress trailing
                            # re-emits / degeneration once the answer is delivered.
                            if not expecting_answer["v"]:
                                continue
                            if ev[0] == "done":
                                expecting_answer["v"] = False
                                query_done_event.set()  # unblock ingestion (pacing)
                            await self._emit_demux_event(websocket, ev)
                except Exception:
                    logger.exception("Persistent session epoch %d failed", epoch)
                    await self._send_error(websocket, "Persistent session epoch failed")
                    break
                epoch += 1
            await websocket.send_json({"type": "session.done"})
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass

    @staticmethod
    def _text_output_state(output: OmniRequestOutput) -> tuple[str, Any, int] | None:
        """Pull (cumulative_text, finish_reason, n_tokens) from a text output."""
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
        return (text, finish_reason, ntok)

    async def _emit_demux_event(self, websocket: WebSocket, ev: tuple) -> None:
        """Translate a demux event into a WebSocket response message."""
        kind = ev[0]
        if kind == "start":
            await websocket.send_json({"type": "response.start"})
        elif kind == "delta":
            await websocket.send_json({"type": "response.text.delta", "delta": ev[1]})
        elif kind == "done":
            await websocket.send_json({"type": "response.text.done", "text": ev[1]})

    async def _receive_config(self, websocket: WebSocket) -> StreamingVideoSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type') if isinstance(msg, dict) else type(msg).__name__}",
            )
            return None

        config_data = {k: v for k, v in msg.items() if k != "type"}
        alias_map = {
            "num_sample_frames": "num_frames",
            "evs_enabled": "enable_frame_filter",
            "evs_threshold": "frame_filter_threshold",
        }
        for old_key, new_key in alias_map.items():
            if old_key in config_data and new_key not in config_data:
                config_data[new_key] = config_data[old_key]

        try:
            config = StreamingVideoSessionConfig(**config_data)
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _process_query(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        sink_frames: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> None:
        """Build prompt, run inference, stream text + audio response."""

        if self._engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

        await self._process_query_engine(
            websocket,
            config,
            frame_buffer,
            sink_frames,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
        )

    # ------------------------------------------------------------------
    # Engine-client path (async_chunk audio streaming)
    # ------------------------------------------------------------------

    async def _process_query_engine(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        sink_frames: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> None:
        """Direct engine_client.generate() path for async_chunk audio."""
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        messages, user_message = self._build_messages(
            config,
            frame_buffer,
            sink_frames,
            audio_buffer,
            message_history,
            query_text,
            prewarmed_frames,
        )

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        if config.use_audio_in_video and len(audio_buffer) > 0:
            request_kwargs["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
            }
        if config.sampling_params_list:
            request_kwargs["sampling_params_list"] = config.sampling_params_list

        try:
            chat_request = ChatCompletionRequest(**request_kwargs)
        except Exception as e:
            await self._send_error(websocket, f"Failed to build request: {e}")
            return

        try:
            engine_prompt = await self._preprocess_to_engine_prompt(chat_request)
        except Exception as e:
            await self._send_error(websocket, f"Preprocess failed: {e}")
            return

        await websocket.send_json({"type": "response.start"})
        text_parts: list[str] = []
        text_done_sent = False
        audio_chunk_count = 0
        # Number of per-step tensors in OmniRequestOutput.audio_data already
        # drained. Used by the fast path to skip already-emitted history.
        audio_chunks_drained = 0
        previous_text = ""
        interrupted = False
        t_start = _time.monotonic()
        t_first_text = None
        t_first_audio = None

        # Wire-level async-chunk switch. "off" means
        # buffer all deltas server-side and flush once at the end; the engine
        # pipeline still overlaps internally.
        async_chunk_mode = video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK
        streaming = async_chunk_mode == "on"
        audio_tail_tensors: list[Any] = []

        try:
            result_gen = self._engine_client.generate(
                prompt=engine_prompt,
                request_id=request_id,
                output_modalities=config.modalities,
            )

            async for output in result_gen:
                # Soft interrupt: drain without sending
                if interrupt_event.is_set():
                    if not interrupted:
                        logger.info("Generation interrupted — draining")
                        interrupted = True
                    continue

                if not isinstance(output, OmniRequestOutput):
                    continue

                out_type = getattr(output, "final_output_type", "text")

                if out_type == "audio":
                    if streaming and not text_done_sent:
                        full_text = "".join(text_parts)
                        await websocket.send_json({"type": "response.text.done", "text": full_text})
                        text_done_sent = True

                    if t_first_audio is None:
                        t_first_audio = _time.monotonic()
                    audio_chunk_count += 1
                    if streaming:
                        b64, audio_chunks_drained = self._extract_audio_delta_b64(
                            output,
                            audio_chunks_drained,
                        )
                        if b64:
                            await websocket.send_json(
                                {
                                    "type": "response.audio.delta",
                                    "data": b64,
                                    "format": "wav",
                                }
                            )
                    else:
                        audio_data = self._get_audio_data(output)
                        if audio_data is not None:
                            if isinstance(audio_data, list):
                                audio_tail_tensors = list(audio_data)
                            else:
                                audio_tail_tensors = [audio_data]
                else:
                    delta_text, previous_text = self._extract_text_delta(
                        output,
                        previous_text,
                    )
                    if delta_text:
                        if t_first_text is None:
                            t_first_text = _time.monotonic()
                        text_parts.append(delta_text)
                        if streaming:
                            await websocket.send_json({"type": "response.text.delta", "delta": delta_text})

            if not text_done_sent:
                full_text = "".join(text_parts)
                await websocket.send_json({"type": "response.text.done", "text": full_text})
                text_done_sent = True

            if not streaming and audio_tail_tensors:
                try:
                    coalesced = (
                        audio_tail_tensors[0] if len(audio_tail_tensors) == 1 else torch.cat(audio_tail_tensors, dim=-1)
                    )
                    tail_np = self._tensor_to_1d_np(coalesced)
                    b64, _ = self._encode_tail(
                        tail_np,
                        0,
                        new_drained=len(audio_tail_tensors),
                        is_first=True,
                    )
                    if b64:
                        await websocket.send_json(
                            {
                                "type": "response.audio.delta",
                                "data": b64,
                                "format": "wav",
                            }
                        )
                except Exception:
                    logger.exception("Failed to coalesce off-path audio")

            if audio_chunk_count > 0:
                await websocket.send_json({"type": "response.audio.done"})

            response_text = "".join(text_parts)
            message_history.append(user_message)
            message_history.append({"role": "assistant", "content": response_text})

            t_end = _time.monotonic()
            logger.info(
                "[TIMING] mode=%s total=%.2fs first_text=%.2fs first_audio=%.2fs audio_chunks=%d",
                async_chunk_mode,
                t_end - t_start,
                (t_first_text - t_start) if t_first_text else -1,
                (t_first_audio - t_start) if t_first_audio else -1,
                audio_chunk_count,
            )

        except Exception:
            logger.exception("Engine query failed")
            await self._send_error(websocket, "Query processing failed")

        if not text_done_sent:
            full_text = "".join(text_parts)
            await websocket.send_json({"type": "response.text.done", "text": full_text})

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        sink_frames: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build OpenAI-style messages list and the current user message.

        Returns (messages, user_message).
        """
        # Stride sampling (index 0 anchor, last slot = newest). Covers full buffer + stable mm_hash.
        n_buf = len(frame_buffer)
        if n_buf <= config.num_frames:
            recent = list(frame_buffer)
        else:
            stride = max(1, n_buf // config.num_frames)
            idx = [i * stride for i in range(config.num_frames - 1)] + [n_buf - 1]
            recent = [frame_buffer[i] for i in idx]

        # B' sink: prepend the pinned opening frames, then the recent window,
        # de-duplicated (order-preserving) so a frame that is both a sink frame
        # and a recent frame is sent only once (also avoids duplicate mm uuids).
        seen: set[str] = set()
        frames: list[str] = []
        for fb in (sink_frames or []) + recent:
            if fb not in seen:
                seen.add(fb)
                frames.append(fb)

        # Prefer prewarmed PIL + uuid so mm_cache can dedupe by hash.
        prewarmed = prewarmed_frames or {}
        user_content: list[dict] = []
        for frame_b64 in frames:
            cached = prewarmed.get(frame_b64)
            if cached is _BAD_FRAME:
                continue
            if cached is not None:
                pil, pil_uuid = cached
                user_content.append(
                    {
                        "type": "image_pil",
                        "image_pil": pil,
                        "uuid": pil_uuid,
                    }
                )
            else:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    }
                )

        if len(audio_buffer) > 0:
            wav_b64 = self._pcm_to_wav_b64(bytes(audio_buffer))
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_b64,
                        "format": "wav",
                    },
                }
            )

        if query_text:
            user_content.append({"type": "text", "text": query_text})

        user_message: dict[str, Any] = {"role": "user", "content": user_content}

        messages: list[dict[str, Any]] = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})

        # Add text-only history (strip images/audio from old turns).
        # Keep only the last turn (2 messages) to keep prompt short
        # enough for single-step mm_encoder scheduling.  When prompt
        # exceeds ~50 tokens, the V1 scheduler splits mm_encoder and
        # text prefill, causing incomplete thinker embeddings and
        # garbled audio.
        recent_history = message_history[-2:] if len(message_history) > 2 else message_history
        for hist_msg in recent_history:
            messages.append(self._text_only_message(hist_msg))

        messages.append(user_message)

        return messages, user_message

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pcm_to_wav_b64(pcm_data: bytes, sample_rate: int = 16000) -> str:
        """Wrap raw PCM16 mono in a WAV container and return base64."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return base64.b64encode(buf.getvalue()).decode()

    @classmethod
    def _extract_audio_delta_b64(
        cls,
        result: OmniRequestOutput,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Return (base64 WAV of new samples, updated chunks_drained).

        `chunks_drained` is the number of per-step tensors in
        ``audio_data`` that have already been emitted. Each engine step appends
        one tensor, so new samples are ``audio_data[chunks_drained:]`` — no
        matter how many steps accumulated between reads (handles backpressure
        cleanly, unlike a simple ``audio_data[-1]``).

        Two paths, selected at runtime by ``VLLM_VIDEO_AUDIO_DELTA_MODE``:
          * fast — only D2H the new tail. Per-call cost ∝ new chunks.
          * slow — full cat + D2H each call. Per-call cost ∝ total history.
                   Retained for A/B; remove once downstream callers confirm.
        """
        audio_data = cls._get_audio_data(result)
        if audio_data is None:
            return None, chunks_drained

        if video_stream_envs.VLLM_VIDEO_AUDIO_DELTA_MODE == "slow":
            return cls._delta_slow(audio_data, chunks_drained)
        return cls._delta_fast(audio_data, chunks_drained)

    @staticmethod
    def _get_audio_data(result: OmniRequestOutput):
        """Navigate OmniRequestOutput → multimodal_output['audio']. None on miss."""
        request_output = getattr(result, "request_output", None)
        if request_output is None:
            return None
        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return None
        mm_output = getattr(outputs[0], "multimodal_output", None)
        if not isinstance(mm_output, dict):
            return None
        return mm_output.get("audio")

    @classmethod
    def _delta_fast(
        cls,
        audio_data,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Emit only tensors appended since the last call."""
        # Single tensor: output_processor hands us one tensor before it becomes a
        # list (see output_processor.py:89). Treat it as chunk #0.
        if not isinstance(audio_data, list):
            if chunks_drained >= 1:
                return None, chunks_drained
            tail_np = cls._tensor_to_1d_np(audio_data)
            return cls._encode_tail(tail_np, chunks_drained, new_drained=1, is_first=True)

        n = len(audio_data)
        if n <= chunks_drained:
            return None, chunks_drained

        new_chunks = audio_data[chunks_drained:]
        tail = new_chunks[0] if len(new_chunks) == 1 else torch.cat(new_chunks, dim=-1)
        tail_np = cls._tensor_to_1d_np(tail)
        return cls._encode_tail(tail_np, chunks_drained, new_drained=n, is_first=(chunks_drained == 0))

    @classmethod
    def _delta_slow(
        cls,
        audio_data,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Pre-fix behaviour: concat everything each call and slice on CPU."""
        if isinstance(audio_data, list):
            if not audio_data:
                return None, chunks_drained
            audio_tensor = torch.cat(audio_data, dim=-1)
            new_drained = len(audio_data)
        else:
            audio_tensor = audio_data
            new_drained = 1

        full_np = cls._tensor_to_1d_np(audio_tensor)
        if full_np is None:
            return None, chunks_drained
        # chunks_drained doesn't map directly to sample offset without tracking
        # per-chunk lengths, so we re-derive: replay the tail that corresponds
        # to chunks appended since last call by slicing off the part produced
        # by the already-drained prefix. For slow path this is intentionally
        # wasteful — the point is to reproduce the pre-fix hot loop.
        if chunks_drained == 0:
            tail_np = full_np
        else:
            # Recover prefix length by re-concatenating the already-drained
            # prefix tensors (cost intentionally identical to the baseline
            # implementation this was lifted from).
            if isinstance(audio_data, list) and chunks_drained < len(audio_data):
                prefix_len = sum(int(t.shape[-1]) for t in audio_data[:chunks_drained])
                tail_np = full_np[prefix_len:]
            else:
                tail_np = full_np[0:0]
        return cls._encode_tail(tail_np, chunks_drained, new_drained=new_drained, is_first=(chunks_drained == 0))

    @classmethod
    def _encode_tail(
        cls,
        tail_np,
        old_drained: int,
        *,
        new_drained: int,
        is_first: bool,
    ) -> tuple[str | None, int]:
        """Strip the CausalConv leading artifact on first emit, then b64-encode."""
        if tail_np is None or len(tail_np) == 0:
            return None, new_drained
        if is_first and len(tail_np) > _CODEC_FRAME_SAMPLES * 2:
            tail_np = tail_np[_CODEC_FRAME_SAMPLES:]
        if len(tail_np) == 0:
            return None, new_drained
        try:
            return cls._encode_audio_wav_b64(tail_np), new_drained
        except Exception:
            logger.exception("Failed to encode audio delta WAV")
            return None, old_drained

    @staticmethod
    def _tensor_to_1d_np(t):
        """Tensor → flat float32 numpy on CPU. None on failure."""
        if t is None or not hasattr(t, "float"):
            return None
        arr = t.float().detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr

    @staticmethod
    def _encode_audio_wav_b64(audio_np) -> str:
        """Encode numpy float32 audio to base64 WAV (24kHz)."""
        from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
        from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

        audio_obj = CreateAudio(
            audio_tensor=audio_np,
            sample_rate=24000,
            response_format="wav",
            speed=1.0,
            stream_format="audio",
            base64_encode=True,
        )
        mixin = AudioMixin()
        resp = mixin.create_audio(audio_obj)
        return resp.audio_data

    @staticmethod
    def _extract_text_delta(
        result: OmniRequestOutput,
        previous_text: str,
    ) -> tuple[str, str]:
        """Extract incremental text delta from OmniRequestOutput."""
        if result.final_output_type != "text":
            return "", previous_text

        request_output = getattr(result, "request_output", None)
        if request_output is None:
            return "", previous_text

        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return "", previous_text

        text = getattr(outputs[0], "text", None)
        if not isinstance(text, str) or not text:
            return "", previous_text

        if text.startswith(previous_text):
            return text[len(previous_text) :], text
        return text, text

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    async def _preprocess_to_engine_prompt(self, request) -> Any:
        """Use the chat handler's preprocessing to build an engine prompt."""
        handler = self._chat_service
        renderer = handler.renderer

        _conversation, engine_prompts = await handler._preprocess_chat(
            request,
            request.messages,
            default_template=getattr(request, "chat_template", None) or handler.chat_template,
            default_template_content_format=handler.chat_template_content_format,
            renderer=renderer,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            add_special_tokens=request.add_special_tokens,
        )
        return engine_prompts[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    _text_only_message = staticmethod(text_only_message)

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass
