# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the serving-layer streaming video WebSocket handler."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
from typing import Any

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai import serving_video_stream, video_stream_envs
from vllm_omni.entrypoints.openai.serving_video_stream import (
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
    _TextStreamDemux,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_jpeg(r: int = 128, g: int = 128, b: int = 128) -> bytes:
    img = Image.new("RGB", (64, 64), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _text_result(text: str) -> OmniRequestOutput:
    class Output:
        pass

    class RequestOutput:
        pass

    output = Output()
    output.text = text
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput(final_output_type="text", request_output=request_output)


def _audio_result(audio_data: Any) -> OmniRequestOutput:
    class Output:
        pass

    class RequestOutput:
        pass

    output = Output()
    output.multimodal_output = {"audio": audio_data}
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput(final_output_type="audio", request_output=request_output)


class MockWebSocket:
    def __init__(self, messages: list[str] | None = None):
        self._messages = list(messages or [])
        self._idx = 0
        self.accepted = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        if self._idx >= len(self._messages):
            await asyncio.sleep(999)
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send_json(self, data: dict[str, Any]):
        self.sent.append(data)


class TimedWebSocket:
    def __init__(self):
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self.accepted = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        return await self._q.get()

    async def send_json(self, data: dict[str, Any]):
        self.sent.append(data)

    def put(self, msg: dict[str, Any]):
        self._q.put_nowait(json.dumps(msg))

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent]


def test_api_server_registers_video_stream_route():
    from vllm_omni.entrypoints.openai.api_server import router

    assert any(getattr(route, "path", None) == "/v1/video/chat/stream" for route in router.routes)


@pytest.mark.asyncio
async def test_receive_config_accepts_client_legacy_aliases():
    ws = MockWebSocket(
        [
            json.dumps(
                {
                    "type": "session.config",
                    "model": "test",
                    "num_sample_frames": 7,
                    "evs_enabled": False,
                    "evs_threshold": 0.87,
                }
            )
        ]
    )
    handler = OmniStreamingVideoHandler(chat_service=object())

    config = await handler._receive_config(ws)

    assert config is not None
    assert config.num_frames == 7
    assert config.enable_frame_filter is False
    assert config.frame_filter_threshold == 0.87


@pytest.mark.asyncio
async def test_audio_in_video_sets_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(OmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text", "audio"], use_audio_in_video=True)

    await handler._process_query_engine(
        ws,
        config,
        [_b64(_make_jpeg())],
        [],
        bytearray(b"\x00\x00"),
        [],
        "what is happening?",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs == {"use_audio_in_video": True}


@pytest.mark.asyncio
async def test_audio_in_video_disabled_omits_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(OmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text", "audio"], use_audio_in_video=False)

    await handler._process_query_engine(
        ws,
        config,
        [_b64(_make_jpeg())],
        [],
        bytearray(b"\x00\x00"),
        [],
        "what is happening?",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs is None


@pytest.mark.asyncio
async def test_query_inline_audio_data_sets_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(OmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket(
        [
            json.dumps({"type": "session.config", "model": "test"}),
            json.dumps({"type": "video.frame", "data": _b64(_make_jpeg())}),
            json.dumps(
                {
                    "type": "video.query",
                    "text": "describe",
                    "audio_data": _b64(b"\x00\x00"),
                }
            ),
            json.dumps({"type": "video.done"}),
        ]
    )
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine(), idle_timeout=2.0)

    await handler.handle_session(ws)

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs == {"use_audio_in_video": True}
    assert "session.done" in [m.get("type") for m in ws.sent]


def test_audio_delta_mode_is_read_by_serving_code_at_runtime(monkeypatch):
    handler = OmniStreamingVideoHandler(chat_service=object())
    result = _audio_result([object()])

    monkeypatch.setattr(
        OmniStreamingVideoHandler,
        "_delta_fast",
        classmethod(lambda cls, audio_data, chunks_drained: ("fast-path", chunks_drained)),
    )
    monkeypatch.setattr(
        OmniStreamingVideoHandler,
        "_delta_slow",
        classmethod(lambda cls, audio_data, chunks_drained: ("slow-path", chunks_drained)),
    )

    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", "fast")
    assert handler._extract_audio_delta_b64(result, 0)[0] == "fast-path"

    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", "slow")
    assert handler._extract_audio_delta_b64(result, 0)[0] == "slow-path"


def test_video_stream_envs_strip_and_warn_once_per_invalid_value(monkeypatch):
    warnings = []

    video_stream_envs._warned_invalid_envs.clear()
    try:
        monkeypatch.setattr(
            video_stream_envs.logger,
            "warning",
            lambda message, *args, **_kwargs: warnings.append((message, args)),
        )

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", " off ")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "off"
        assert not warnings

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "bad")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert len(warnings) == 1

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "still_bad")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert len(warnings) == 2
    finally:
        video_stream_envs._warned_invalid_envs.clear()


@pytest.mark.asyncio
async def test_async_chunk_mode_is_read_by_engine_path_at_runtime(monkeypatch):
    class TextEngine:
        def generate(self, **_kwargs):
            async def _gen():
                yield _text_result("hello")

            return _gen()

    class CapturingHandler(OmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "x"}

    handler = CapturingHandler(chat_service=object(), engine_client=TextEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text"])

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "on")
    ws_on = MockWebSocket()
    await handler._process_query_engine(
        ws_on,
        config,
        [_b64(_make_jpeg())],
        [],
        bytearray(),
        [],
        "describe",
        "req-on",
        asyncio.Event(),
        {},
    )
    assert {"type": "response.text.delta", "delta": "hello"} in ws_on.sent

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "off")
    ws_off = MockWebSocket()
    await handler._process_query_engine(
        ws_off,
        config,
        [_b64(_make_jpeg())],
        [],
        bytearray(),
        [],
        "describe",
        "req-off",
        asyncio.Event(),
        {},
    )
    assert {"type": "response.text.done", "text": "hello"} in ws_off.sent
    assert not any(m.get("type") == "response.text.delta" for m in ws_off.sent)


@pytest.mark.asyncio
async def test_query_without_engine_client_sends_error():
    ws = MockWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), engine_client=None)

    await handler._process_query(
        ws,
        StreamingVideoSessionConfig(model="test"),
        [],
        [],
        bytearray(),
        [],
        "describe",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert {"type": "error", "message": "Streaming video requires an engine client"} in ws.sent


@pytest.mark.asyncio
async def test_new_query_cancels_in_flight_query():
    query_started = asyncio.Event()
    query_cancelled = asyncio.Event()
    calls = 0

    class BlockingHandler(OmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                return
            query_started.set()
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                query_cancelled.set()
                raise

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    ws.put({"type": "video.query", "text": "interrupt"})
    await asyncio.wait_for(query_cancelled.wait(), timeout=2.0)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=2.0)
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_video_done_waits_for_in_flight_query():
    query_started = asyncio.Event()
    allow_finish = asyncio.Event()
    query_finished = asyncio.Event()

    class BlockingHandler(OmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            query_started.set()
            await allow_finish.wait()
            query_finished.set()

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    ws.put({"type": "video.done"})
    await asyncio.sleep(0.05)
    assert not task.done()
    assert not query_finished.is_set()

    allow_finish.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert query_finished.is_set()
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_frame_prewarm_does_not_block_following_query(monkeypatch):
    decode_started = threading.Event()
    release_decode = threading.Event()
    query_started = asyncio.Event()

    def blocked_decode(raw_bytes: bytes):
        decode_started.set()
        release_decode.wait(timeout=2.0)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    class BlockingHandler(OmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            query_started.set()

    monkeypatch.setattr(serving_video_stream, "_decode_frame_bytes", blocked_decode)

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})

    for _ in range(100):
        if decode_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert decode_started.is_set()

    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    release_decode.set()
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_client_cannot_send_internal_frame_decode_failed_message():
    captured_frames: list[list[str]] = []
    frame = _b64(_make_jpeg())

    class CapturingHandler(OmniStreamingVideoHandler):
        async def _process_query(
            self,
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
        ):
            captured_frames.append(list(frame_buffer))

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": frame})
    await asyncio.sleep(0)
    ws.put({"type": "_internal.frame_decode_failed", "b64": frame})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Unknown type: _internal.frame_decode_failed"} in ws.sent
    assert captured_frames == [[frame]]


@pytest.mark.asyncio
async def test_failed_frame_prewarm_removes_frame_before_query():
    ws = TimedWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test", "enable_frame_filter": False})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(b"not-a-jpeg")})

    for _ in range(100):
        if any(m.get("message") == "Frame decode failed" for m in ws.sent):
            break
        await asyncio.sleep(0.01)

    assert {"type": "error", "message": "Frame decode failed"} in ws.sent

    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "No frames buffered"} in ws.sent


@pytest.mark.asyncio
async def test_frame_filter_error_sends_invalid_image(monkeypatch):
    def fail_should_retain(self, frame_jpeg):
        raise ValueError("decode failed")

    monkeypatch.setattr(serving_video_stream.FrameSimilarityFilter, "should_retain", fail_should_retain)

    ws = TimedWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Invalid image data"} in ws.sent
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_audio_buffer_overflow_clears_buffer_before_query(monkeypatch):
    captured_audio_lengths: list[int] = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(OmniStreamingVideoHandler):
        async def _process_query_engine(
            self,
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
        ):
            captured_audio_lengths.append(len(audio_buffer))

    monkeypatch.setattr(serving_video_stream, "_MAX_AUDIO_BUFFER_BYTES", 4)

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "audio.chunk", "data": _b64(b"1234")})
    await asyncio.sleep(0)
    ws.put({"type": "audio.chunk", "data": _b64(b"5")})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Audio buffer overflow"} in ws.sent
    assert captured_audio_lengths == [0]


def test_build_messages_keeps_recent_history_text_only():
    handler = OmniStreamingVideoHandler(chat_service=object())
    old_frame = _b64(_make_jpeg(1, 2, 3))
    current_frame = _b64(_make_jpeg(4, 5, 6))
    history = [
        {"role": "user", "content": [{"type": "text", "text": "old question"}]},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{old_frame}"}},
                {"type": "input_audio", "input_audio": {"data": "ignored", "format": "wav"}},
                {"type": "text", "text": "recent question"},
            ],
        },
        {"role": "assistant", "content": "recent answer"},
    ]

    messages, user_message = handler._build_messages(
        StreamingVideoSessionConfig(model="test", num_frames=1),
        [current_frame],
        [],
        bytearray(),
        history,
        "current question",
        {},
    )

    assert messages[0] == {"role": "user", "content": "recent question"}
    assert messages[1] == {"role": "assistant", "content": "recent answer"}
    assert messages[2] == user_message
    assert user_message["content"][-1] == {"type": "text", "text": "current question"}


def _img_urls(user_message):
    return [
        c["image_url"]["url"].split(",", 1)[1]
        for c in user_message["content"]
        if c["type"] == "image_url"
    ]


def test_build_messages_prepends_sink_frames():
    """B': pinned opening (sink) frames are prepended before the recent window."""
    handler = OmniStreamingVideoHandler(chat_service=object())
    sink0 = _b64(_make_jpeg(1, 1, 1))
    sink1 = _b64(_make_jpeg(2, 2, 2))
    recent = _b64(_make_jpeg(9, 9, 9))
    _, user_message = handler._build_messages(
        StreamingVideoSessionConfig(model="test", num_frames=1, sink_frames=2),
        [recent],
        [sink0, sink1],
        bytearray(),
        [],
        "q",
        {},
    )
    assert _img_urls(user_message) == [sink0, sink1, recent]


def test_build_messages_sink_dedups_overlap_with_recent():
    """A frame that is both a sink frame and in the recent window is sent once."""
    handler = OmniStreamingVideoHandler(chat_service=object())
    f0 = _b64(_make_jpeg(1, 1, 1))
    f1 = _b64(_make_jpeg(2, 2, 2))
    _, user_message = handler._build_messages(
        StreamingVideoSessionConfig(model="test", num_frames=2, sink_frames=2),
        [f0, f1],
        [f0, f1],
        bytearray(),
        [],
        "q",
        {},
    )
    assert _img_urls(user_message) == [f0, f1]


def test_build_messages_no_sink_is_unchanged():
    """sink_frames=0 (default) preserves the current windowed behavior."""
    handler = OmniStreamingVideoHandler(chat_service=object())
    recent = _b64(_make_jpeg(7, 7, 7))
    _, user_message = handler._build_messages(
        StreamingVideoSessionConfig(model="test", num_frames=1),
        [recent],
        [],
        bytearray(),
        [],
        "q",
        {},
    )
    assert _img_urls(user_message) == [recent]


# ----------------------------------------------------------------------
# Persistent streaming-LLM session (config.persistent=True)
# ----------------------------------------------------------------------


def _omni_text(cum: str, finish_reason: Any = None, ntok: int | None = None) -> OmniRequestOutput:
    """Text OmniRequestOutput carrying cumulative text + finish_reason + token_ids."""

    class Output:
        pass

    class RequestOutput:
        pass

    output = Output()
    output.text = cum
    output.finish_reason = finish_reason
    output.token_ids = list(range(ntok if ntok is not None else len(cum.split())))
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput(final_output_type="text", request_output=request_output)


def _feed_all(demux: _TextStreamDemux, seq: list[tuple]) -> list[tuple]:
    events: list[tuple] = []
    for cum, finish_reason, ntok in seq:
        events.extend(demux.feed(cum, finish_reason, ntok))
    return events


def test_demux_ignores_empty_input_chunk_outputs():
    """Input-only frame chunks emit empty-text outputs — never surfaced."""
    demux = _TextStreamDemux()
    assert _feed_all(demux, [("", "length", 5), ("", "length", 9)]) == []


def test_demux_streams_query_token_by_token():
    """A query streams as incremental deltas, then done with the full text."""
    demux = _TextStreamDemux()
    events = _feed_all(
        demux,
        [
            ("The", None, 1),
            ("The cat", None, 2),
            ("The cat sat", "stop", 3),
        ],
    )
    assert events == [
        ("start",),
        ("delta", "The"),
        ("delta", " cat"),
        ("delta", " sat"),
        ("done", "The cat sat"),
    ]


def test_demux_query_in_single_finished_output():
    """A query answer that arrives fully in one finished output."""
    demux = _TextStreamDemux()
    events = _feed_all(demux, [("hello there", "stop", 2)])
    assert events == [("start",), ("delta", "hello there"), ("done", "hello there")]


def test_demux_dedups_reemit_after_done():
    """A finished answer that the engine re-yields is not streamed again."""
    demux = _TextStreamDemux()
    events = _feed_all(
        demux,
        [
            ("a", None, 1),
            ("a b", "stop", 2),
            ("a b", "stop", 2),  # re-emit
            ("a b", "stop", 2),  # re-emit
        ],
    )
    assert events == [("start",), ("delta", "a"), ("delta", " b"), ("done", "a b")]


def test_demux_empty_throwaways_and_reemits_between_queries():
    """Empty input-chunk outputs and a re-yielded finished answer between two queries
    must not leak; each query streams exactly once."""
    demux = _TextStreamDemux()
    events = _feed_all(
        demux,
        [
            ("(a) X", None, 2),
            ("(a) X done", "stop", 4),  # query 1 streams to done
            ("", "length", 6),  # empty throwaway
            ("(a) X done", "stop", 4),  # re-emit of the delivered answer -> ignored
            ("", "length", 7),  # empty throwaway
            ("(a) Y", None, 2),
            ("(a) Y done", "stop", 4),  # query 2 streams
        ],
    )
    assert events == [
        ("start",),
        ("delta", "(a) X"),
        ("delta", " done"),
        ("done", "(a) X done"),
        ("start",),
        ("delta", "(a) Y"),
        ("delta", " done"),
        ("done", "(a) Y done"),
    ]


def test_persistent_config_defaults_off():
    config = StreamingVideoSessionConfig(model="test")
    assert config.persistent is False
    assert config.refresh_at_position == 60000


@pytest.mark.asyncio
async def test_persistent_config_dispatches_to_persistent_session():
    called = {}

    class DispatchHandler(OmniStreamingVideoHandler):
        async def _run_persistent_session(self, websocket, config):
            called["persistent"] = config.persistent
            await websocket.send_json({"type": "session.done"})

    ws = MockWebSocket([json.dumps({"type": "session.config", "model": "test", "persistent": True})])
    handler = DispatchHandler(chat_service=object(), engine_client=object())

    await handler.handle_session(ws)

    assert called.get("persistent") is True
    assert any(m.get("type") == "session.done" for m in ws.sent)


class _FakePersistentEngine:
    """Drives the handler's per-epoch input stream and emits simulated outputs.

    Consumes the ``prompt`` async generator (which pulls frames/queries from the
    WS reader), recording each chunk. Query chunks (``max_tokens > 1``) emit a
    cumulative multi-token answer; input-only chunks emit an empty-text throwaway.
    """

    def __init__(self, answer_tokens: list[str]):
        self._answer = answer_tokens
        self.epochs = 0
        self.chunks: list[dict[str, Any]] = []

    def generate(self, *, prompt, sampling_params=None, request_id="", output_modalities=None):
        epoch = self.epochs
        self.epochs += 1
        chunks = self.chunks
        answer = self._answer

        async def _run():
            async for chunk in prompt:
                sp = chunk.sampling_params
                chunks.append(
                    {
                        "epoch": epoch,
                        "text": chunk.prompt["prompt"],
                        "has_mm": "multi_modal_data" in chunk.prompt,
                        "max_tokens": sp.max_tokens,
                    }
                )
                if sp.max_tokens > 1:  # query chunk -> cumulative answer
                    for i in range(len(answer)):
                        cum = " ".join(answer[: i + 1])
                        fr = "stop" if i == len(answer) - 1 else None
                        yield _omni_text(cum, fr, i + 1)
                else:  # input-only -> empty-text throwaway (as the real engine emits)
                    yield _omni_text("", "length", 1)

        return _run()


@pytest.mark.asyncio
async def test_persistent_session_streams_query_response():
    engine = _FakePersistentEngine(["The", "cat", "sat"])
    ws = TimedWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "persistent": True,
            "sink_frames": 1,
            "num_frames": 2,
            "refresh_at_position": 100000,
        }
    )
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(1, 1, 1))})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(2, 2, 2))})
    await asyncio.sleep(0.05)
    ws.put({"type": "video.query", "text": "what is happening?"})
    await asyncio.sleep(0.05)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=3.0)

    types = ws.sent_types()
    assert "response.start" in types
    done = [m for m in ws.sent if m.get("type") == "response.text.done"]
    assert done and done[-1]["text"] == "The cat sat"
    assert "session.done" in types
    assert engine.epochs == 1  # no refresh

    # First chunk is the epoch-0 seed: chat preamble + frames, input-only.
    assert engine.chunks[0]["text"].startswith("<|im_start|>system")
    assert engine.chunks[0]["has_mm"] is True
    assert engine.chunks[0]["max_tokens"] == 1
    # The query opened the assistant turn with no video.
    qchunks = [c for c in engine.chunks if c["max_tokens"] > 1]
    assert qchunks and "<|im_start|>assistant" in qchunks[0]["text"]
    assert qchunks[0]["has_mm"] is False


@pytest.mark.asyncio
async def test_persistent_session_refreshes_and_reseeds_opening(monkeypatch):
    # Scale the per-chunk position estimate up so a couple of frames cross the
    # (config-minimum) refresh threshold of 1024 -> forces a refresh in-test.
    monkeypatch.setattr(serving_video_stream, "_PERSIST_EST_POS_PER_CHUNK", 700)
    engine = _FakePersistentEngine(["A", "B", "C"])
    ws = TimedWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "persistent": True,
            "sink_frames": 1,
            "num_frames": 2,
            "refresh_at_position": 1024,  # min allowed; ~2 chunks/epoch -> refresh
        }
    )
    await asyncio.sleep(0)
    for shade in range(1, 7):  # 6 distinct frames
        ws.put({"type": "video.frame", "data": _b64(_make_jpeg(shade, shade, shade))})
    await asyncio.sleep(0.1)
    ws.put({"type": "video.query", "text": "what is happening?"})
    await asyncio.sleep(0.1)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=5.0)

    assert engine.epochs >= 2  # at least one refresh occurred
    # A refresh re-seed: an epoch>0 first carries the chat preamble + the pinned
    # opening frames (so opening recall is preserved across the position wall).
    reseeds = [
        c for c in engine.chunks if c["epoch"] >= 1 and c["text"].startswith("<|im_start|>system") and c["has_mm"]
    ]
    assert reseeds
    # The query is still answered (in whichever epoch reached it).
    done = [m for m in ws.sent if m.get("type") == "response.text.done"]
    assert done and done[-1]["text"] == "A B C"
    assert "session.done" in ws.sent_types()
