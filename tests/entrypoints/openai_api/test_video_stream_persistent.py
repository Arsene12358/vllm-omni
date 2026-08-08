# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the persistent streaming-LLM video session.

Covers the ``_TextStreamDemux`` event demultiplexer, the persistent-session
fields on ``StreamingVideoSessionConfig``, and the ``run_persistent_session``
driver (real asyncio, fake engine client + WebSocket).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Any

import pytest
from fastapi import WebSocketDisconnect
from PIL import Image
from pydantic import ValidationError

from vllm_omni.entrypoints.openai import video_stream_persistent
from vllm_omni.entrypoints.openai.serving_video_stream import QwenOmniStreamingVideoHandler
from vllm_omni.entrypoints.openai.video_stream_base import (
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
)
from vllm_omni.entrypoints.openai.video_stream_persistent import (
    _VID_BLOCK,
    _TextStreamDemux,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# _TextStreamDemux
# ---------------------------------------------------------------------------


def test_input_only_outputs_ignored():
    d = _TextStreamDemux()
    assert d.feed("", None, 0) == [] and d.feed("   ", None, 1) == []


def test_delta_growth_and_done():
    d = _TextStreamDemux()
    assert d.feed("Hel", None, 1) == [("start",), ("delta", "Hel")]
    assert d.feed("Hello", None, 2) == [("delta", "lo")]
    assert d.feed("Hello.", "stop", 3) == [("delta", "."), ("done", "Hello.")]


def test_finished_answer_reemit_deduped():
    d = _TextStreamDemux()
    d.feed("Hi", "stop", 1)
    assert d.feed("Hi", None, 1) == []  # engine re-yield of delivered answer


def test_non_prefix_growth_resets_delta():
    d = _TextStreamDemux()
    d.feed("abc", None, 1)
    assert d.feed("xyz", None, 2) == [("delta", "xyz")]


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


# ---------------------------------------------------------------------------
# StreamingVideoSessionConfig persistent-session fields
# ---------------------------------------------------------------------------


def test_persistent_config_defaults_off():
    config = StreamingVideoSessionConfig(model="test")
    assert config.persistent is False
    assert config.refresh_at_position == 60000
    assert config.sink_frames == 0


def test_persistent_config_field_bounds():
    with pytest.raises(ValidationError):
        StreamingVideoSessionConfig(model="test", refresh_at_position=1023)
    with pytest.raises(ValidationError):
        StreamingVideoSessionConfig(model="test", refresh_at_position=262145)
    with pytest.raises(ValidationError):
        StreamingVideoSessionConfig(model="test", sink_frames=-1)
    with pytest.raises(ValidationError):
        StreamingVideoSessionConfig(model="test", sink_frames=65)


def test_engine_rebase_config_defaults_and_validation():
    assert StreamingVideoSessionConfig(model="test").engine_rebase is False
    assert StreamingVideoSessionConfig(model="test", engine_rebase=True).engine_rebase is True
    with pytest.raises(ValidationError):
        StreamingVideoSessionConfig(model="test", engine_rebase="not-a-bool")


# ---------------------------------------------------------------------------
# run_persistent_session driver
# ---------------------------------------------------------------------------

_SEED_PREFIX = (
    "<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, "
    "Alibaba Group, capable of perceiving auditory and visual inputs.<|im_end|>\n<|im_start|>user\n"
)


def _make_jpeg(shade: int = 128) -> bytes:
    img = Image.new("RGB", (64, 64), (shade, shade, shade))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _shade(video, index: int) -> int:
    """Red channel of the first pixel of frame ``index`` in a chunk's video array."""
    return int(video[index, 0, 0, 0])


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


class _AnswerGate:
    """Hold the fake engine just before the final token of an answer."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.release = asyncio.Event()

    async def hold(self) -> None:
        self.reached.set()
        await self.release.wait()


class FakeEngineClient:
    """Fake ``AsyncOmni.generate()`` for the persistent session driver.

    Mirrors the real v0.26.0 streaming-input path: the input generator is
    drained by a *separate* task (as ``AsyncOmni._add_streaming_input_request``
    does) while the result generator pumps outputs, so the driver's query pause
    is observable instead of being an artifact of a serialized fake.

    Query chunks (``max_tokens > 1``) emit a cumulative multi-token answer;
    input-only chunks emit the empty-text throwaway the real engine produces.
    """

    def __init__(
        self,
        answer_tokens: list[str] | None = None,
        *,
        ingest_start: asyncio.Event | None = None,
        answer_gate: _AnswerGate | None = None,
        after_answer: list[tuple[str, Any, int]] | None = None,
        fail: bool = False,
    ) -> None:
        self._answer = answer_tokens or ["ok"]
        self._ingest_start = ingest_start
        self._answer_gate = answer_gate
        self._after_answer = after_answer or []
        self._fail = fail
        self.epochs = 0
        self.aborted = False  # result stream closed before it was drained
        self.calls: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

    def epoch_chunks(self, epoch: int) -> list[dict[str, Any]]:
        return [c for c in self.chunks if c["epoch"] == epoch]

    def generate(self, *, prompt, sampling_params=None, request_id="", output_modalities=None):
        epoch = self.epochs
        self.epochs += 1
        self.calls.append(
            {
                "request_id": request_id,
                "sampling_params": sampling_params,
                "output_modalities": output_modalities,
            }
        )
        outq: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        async def _drain() -> None:
            try:
                if self._ingest_start is not None:
                    await self._ingest_start.wait()
                async for chunk in prompt:
                    sp = chunk.sampling_params
                    video = chunk.prompt.get("multi_modal_data", {}).get("video")
                    self.chunks.append(
                        {
                            "epoch": epoch,
                            "text": chunk.prompt["prompt"],
                            "video": video,
                            "n_frames": 0 if video is None else int(video.shape[0]),
                            "max_tokens": sp.max_tokens,
                            "repetition_penalty": sp.repetition_penalty,
                        }
                    )
                    if sp.max_tokens > 1:  # query chunk -> cumulative answer
                        last = len(self._answer) - 1
                        for i in range(len(self._answer)):
                            cum = " ".join(self._answer[: i + 1])
                            await outq.put((_omni_text(cum, "stop" if i == last else None, i + 1), i == last))
                        for cum, finish_reason, ntok in self._after_answer:
                            await outq.put((_omni_text(cum, finish_reason, ntok), False))
                    else:  # input-only -> empty-text throwaway
                        await outq.put((_omni_text("", "length", 1), False))
            finally:
                await outq.put((sentinel, False))

        async def _run():
            if self._fail:
                raise RuntimeError("engine exploded")
                yield  # pragma: no cover - unreachable, keeps _run an async generator
            task = asyncio.create_task(_drain())
            try:
                while True:
                    output, gated = await outq.get()
                    if output is sentinel:
                        return
                    if gated and self._answer_gate is not None:
                        await self._answer_gate.hold()
                    yield output
            except GeneratorExit:
                # The driver closed the stream mid-epoch; the real AsyncOmni
                # aborts its internal requests on exactly this signal.
                self.aborted = True
                raise
            finally:
                task.cancel()

        return _run()


def _persistent_handler(engine: Any) -> OmniStreamingVideoHandler:
    return QwenOmniStreamingVideoHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)


def _config_msg(**overrides: Any) -> dict[str, Any]:
    msg = {
        "type": "session.config",
        "model": "test",
        "modalities": ["text"],
        "persistent": True,
        "sink_frames": 1,
        "num_frames": 2,
        "refresh_at_position": 100000,
    }
    msg.update(overrides)
    return msg


async def _settle(ws: TimedWebSocket, ticks: int = 60) -> None:
    """Let the session reader drain and decode every queued client message."""
    for _ in range(ticks):
        await asyncio.sleep(0.01)
        if ws._q.empty():
            break
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_persistent_config_dispatches_to_run_persistent_session(monkeypatch):
    called: dict[str, Any] = {}

    async def _fake_driver(handler, websocket, config):
        called["handler"] = handler
        called["persistent"] = config.persistent
        await websocket.send_json({"type": "session.done"})

    monkeypatch.setattr(video_stream_persistent, "run_persistent_session", _fake_driver)

    ws = MockWebSocket([json.dumps({"type": "session.config", "model": "test", "persistent": True})])
    handler = QwenOmniStreamingVideoHandler(chat_service=object(), engine_client=object())

    await handler.handle_session(ws)

    assert called.get("persistent") is True
    assert called.get("handler") is handler
    assert any(m.get("type") == "session.done" for m in ws.sent)


@pytest.mark.asyncio
async def test_non_persistent_config_keeps_windowed_path(monkeypatch):
    async def _boom(handler, websocket, config):  # pragma: no cover - must not run
        raise AssertionError("persistent driver must not run for persistent=False")

    monkeypatch.setattr(video_stream_persistent, "run_persistent_session", _boom)

    ws = MockWebSocket(
        [
            json.dumps({"type": "session.config", "model": "test"}),
            json.dumps({"type": "video.done"}),
        ]
    )
    handler = QwenOmniStreamingVideoHandler(chat_service=object(), engine_client=object())

    await handler.handle_session(ws)

    assert [m.get("type") for m in ws.sent] == ["session.done"]


@pytest.mark.asyncio
async def test_persistent_session_seeds_epoch_zero_and_streams_query_response():
    engine = FakeEngineClient(["The", "cat", "sat"])
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(40))})
    await _settle(ws)
    ws.put({"type": "video.query", "text": "what is happening?"})
    await _settle(ws)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=5.0)

    types = ws.sent_types()
    assert "response.start" in types
    done = [m for m in ws.sent if m.get("type") == "response.text.done"]
    assert done and done[-1]["text"] == "The cat sat"
    assert "session.done" in types
    assert engine.epochs == 1  # no refresh

    # Epoch-0 seed: chat preamble + video block + frames, input-only.
    seed = engine.chunks[0]
    assert seed["text"] == _SEED_PREFIX + _VID_BLOCK
    assert seed["n_frames"] >= 1
    assert seed["max_tokens"] == 1

    # The query closes the user turn, opens the assistant turn, carries no video.
    queries = [c for c in engine.chunks if c["max_tokens"] > 1]
    assert len(queries) == 1
    assert queries[0]["text"] == " what is happening?<|im_end|>\n<|im_start|>assistant\n"
    assert queries[0]["n_frames"] == 0
    assert queries[0]["max_tokens"] == 300
    assert queries[0]["repetition_penalty"] == 1.3

    # generate() is driven with the frozen request-id scheme and text-only outputs.
    call = engine.calls[0]
    assert re.fullmatch(r"vsess-[0-9a-f]{8}-0", call["request_id"])
    assert call["output_modalities"] == ["text"]
    assert call["sampling_params"].max_tokens == 1


@pytest.mark.asyncio
async def test_frames_batch_two_per_chunk_and_query_never_reorders():
    ingest_start = asyncio.Event()
    engine = FakeEngineClient(["ok"], ingest_start=ingest_start)
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    for shade in (10, 40, 70):
        ws.put({"type": "video.frame", "data": _b64(_make_jpeg(shade))})
    ws.put({"type": "video.query", "text": "q"})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(100))})
    # Everything is queued before the engine pulls its first chunk, so the
    # batching/stash behaviour is exercised deterministically.
    await _settle(ws)
    ingest_start.set()
    await _settle(ws)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=5.0)

    shapes = [(c["text"], c["n_frames"], c["max_tokens"]) for c in engine.chunks]
    assert shapes == [
        (_SEED_PREFIX + _VID_BLOCK, 2, 1),  # frames 1-2 batched into the seed
        (_VID_BLOCK, 1, 1),  # frame 3 alone: the query broke the batch
        (" q<|im_end|>\n<|im_start|>assistant\n", 0, 300),  # query stays behind frame 3
        (_VID_BLOCK, 1, 1),  # frame 4 arrived after the query
    ]
    assert _shade(engine.chunks[0]["video"], 0) == pytest.approx(10, abs=6)
    assert _shade(engine.chunks[0]["video"], 1) == pytest.approx(40, abs=6)
    assert _shade(engine.chunks[1]["video"], 0) == pytest.approx(70, abs=6)
    assert _shade(engine.chunks[3]["video"], 0) == pytest.approx(100, abs=6)


@pytest.mark.asyncio
async def test_query_chunk_uses_message_max_tokens():
    engine = FakeEngineClient(["ok"])
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    await _settle(ws)
    ws.put({"type": "video.query", "text": "q", "max_tokens": 17})
    await _settle(ws)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=5.0)

    queries = [c for c in engine.chunks if c["max_tokens"] > 1]
    assert [c["max_tokens"] for c in queries] == [17]


@pytest.mark.asyncio
async def test_ingestion_pauses_until_query_answer_completes():
    gate = _AnswerGate()
    engine = FakeEngineClient(["The", "cat", "sat"], answer_gate=gate)
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    ws.put({"type": "video.query", "text": "q"})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(40))})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(70))})

    await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
    await _settle(ws)
    # Frames queued behind the query must not be ingested while the answer is
    # still decoding — otherwise a session update would truncate it.
    assert [c["max_tokens"] for c in engine.chunks] == [1, 300]

    gate.release.set()
    await _settle(ws)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=5.0)

    assert [c["max_tokens"] for c in engine.chunks] == [1, 300, 1]
    done = [m for m in ws.sent if m.get("type") == "response.text.done"]
    assert done and done[-1]["text"] == "The cat sat"


@pytest.mark.asyncio
async def test_refresh_reseeds_opening_and_recent_with_epoch_request_ids(monkeypatch):
    # Scale the per-chunk position estimate up so a couple of frame chunks cross
    # the (config-minimum) refresh threshold of 1024 -> forces refreshes in-test.
    monkeypatch.setattr(video_stream_persistent, "_PERSIST_EST_POS_PER_CHUNK", 700)
    ingest_start = asyncio.Event()
    engine = FakeEngineClient(["ok"], ingest_start=ingest_start)
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg(refresh_at_position=1024))
    await asyncio.sleep(0)
    for shade in (10, 40, 70, 100, 130, 160):
        ws.put({"type": "video.frame", "data": _b64(_make_jpeg(shade))})
    ws.put({"type": "video.done"})
    await _settle(ws)
    ingest_start.set()

    await asyncio.wait_for(task, timeout=5.0)

    assert engine.epochs == 3
    for i, call in enumerate(engine.calls):  # frozen demo contract: vsess-<8hex>-<epoch>
        assert re.fullmatch(rf"vsess-[0-9a-f]{{8}}-{i}", call["request_id"]), call["request_id"]

    # Epoch 0: seed [10, 40], then ingest [70, 100] -> position wall -> refresh.
    epoch0 = engine.epoch_chunks(0)
    assert [(c["text"], c["n_frames"]) for c in epoch0] == [
        (_SEED_PREFIX + _VID_BLOCK, 2),
        (_VID_BLOCK, 2),
    ]

    # Epoch 1 re-seeds [opening, recent]: the pinned first frame carries the
    # chat preamble, the recent window follows as a bare video block.
    epoch1 = engine.epoch_chunks(1)
    assert [(c["text"], c["n_frames"], c["max_tokens"]) for c in epoch1[:2]] == [
        (_SEED_PREFIX + _VID_BLOCK, 1, 1),
        (_VID_BLOCK, 2, 1),
    ]
    assert _shade(epoch1[0]["video"], 0) == pytest.approx(10, abs=6)  # opening (sink)
    assert _shade(epoch1[1]["video"], 0) == pytest.approx(70, abs=6)  # recent window
    assert _shade(epoch1[1]["video"], 1) == pytest.approx(100, abs=6)

    # Epoch 2 re-seeds the same opening with the newer recent window.
    epoch2 = engine.epoch_chunks(2)
    assert _shade(epoch2[0]["video"], 0) == pytest.approx(10, abs=6)
    assert [_shade(epoch2[1]["video"], i) for i in (0, 1)] == [
        pytest.approx(130, abs=6),
        pytest.approx(160, abs=6),
    ]
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_reseed_alone_does_not_spin_zero_progress_refresh_loop(monkeypatch):
    # One re-seed chunk alone exceeds the refresh threshold. Without the
    # `progressed` gate the epoch loop would refresh forever without ever
    # consuming another event (the session would never finish).
    monkeypatch.setattr(video_stream_persistent, "_PERSIST_EST_POS_PER_CHUNK", 2000)
    ingest_start = asyncio.Event()
    engine = FakeEngineClient(["ok"], ingest_start=ingest_start)
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg(refresh_at_position=1024))
    await asyncio.sleep(0)
    for shade in (10, 40, 70, 100):
        ws.put({"type": "video.frame", "data": _b64(_make_jpeg(shade))})
    ws.put({"type": "video.done"})
    await _settle(ws)
    ingest_start.set()

    await asyncio.wait_for(task, timeout=5.0)

    assert engine.epochs == 2
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_engine_rebase_single_request_survives_refresh_volume(monkeypatch):
    # A frame volume that would force refreshes in fallback mode: 4 chunks at
    # 30000 estimated positions each crosses the default refresh_at_position
    # (60000) twice over. With engine_rebase the engine keeps positions
    # bounded, so the whole session must stay one epoch-0 engine request.
    monkeypatch.setattr(video_stream_persistent, "_PERSIST_EST_POS_PER_CHUNK", 30000)
    warnings: list[str] = []
    monkeypatch.setattr(
        video_stream_persistent.logger,
        "warning",
        lambda message, *args, **_kwargs: warnings.append(message % args if args else message),
    )
    ingest_start = asyncio.Event()
    engine = FakeEngineClient(["ok"], ingest_start=ingest_start)
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    msg = _config_msg(engine_rebase=True)
    del msg["refresh_at_position"]  # left unset -> no "ignored" warning expected
    ws.put(msg)
    await asyncio.sleep(0)
    for shade in (10, 40, 70, 100, 130, 160, 190, 220):
        ws.put({"type": "video.frame", "data": _b64(_make_jpeg(shade))})
    ws.put({"type": "video.done"})
    await _settle(ws)
    ingest_start.set()

    await asyncio.wait_for(task, timeout=5.0)

    # ONE engine request for the session lifetime, epoch pinned to 0.
    assert engine.epochs == 1
    assert len(engine.calls) == 1
    assert re.fullmatch(r"vsess-[0-9a-f]{8}-0", engine.calls[0]["request_id"])

    # Zero re-seed chunks: exactly one seed-prefix chunk (the epoch-0 seed);
    # every later chunk is a bare video block extending the same request.
    shapes = [(c["text"], c["n_frames"], c["max_tokens"]) for c in engine.chunks]
    assert shapes == [
        (_SEED_PREFIX + _VID_BLOCK, 2, 1),
        (_VID_BLOCK, 2, 1),
        (_VID_BLOCK, 2, 1),
        (_VID_BLOCK, 2, 1),
    ]
    assert sum(c["text"].startswith(_SEED_PREFIX) for c in engine.chunks) == 1
    assert "session.done" in ws.sent_types()
    assert not any("refresh_at_position" in w for w in warnings)


@pytest.mark.asyncio
async def test_engine_rebase_warns_once_and_ignores_refresh_at_position(monkeypatch):
    # refresh_at_position explicitly set alongside engine_rebase: one warning,
    # value ignored — the estimated position crosses it and nothing refreshes.
    monkeypatch.setattr(video_stream_persistent, "_PERSIST_EST_POS_PER_CHUNK", 20000)
    warnings: list[str] = []
    monkeypatch.setattr(
        video_stream_persistent.logger,
        "warning",
        lambda message, *args, **_kwargs: warnings.append(message % args if args else message),
    )
    ingest_start = asyncio.Event()
    engine = FakeEngineClient(["ok"], ingest_start=ingest_start)
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg(engine_rebase=True, refresh_at_position=30000))
    await asyncio.sleep(0)
    for shade in (10, 40, 70, 100):
        ws.put({"type": "video.frame", "data": _b64(_make_jpeg(shade))})
    ws.put({"type": "video.done"})
    await _settle(ws)
    ingest_start.set()

    await asyncio.wait_for(task, timeout=5.0)

    ignored = [w for w in warnings if "refresh_at_position" in w]
    assert len(ignored) == 1
    assert "30000" in ignored[0] and "ignored" in ignored[0]
    assert engine.epochs == 1  # no refresh despite crossing the configured position
    assert re.fullmatch(r"vsess-[0-9a-f]{8}-0", engine.calls[0]["request_id"])
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_one_answer_per_query_suppresses_post_answer_reemit():
    engine = FakeEngineClient(
        ["The", "cat", "sat"],
        after_answer=[("The cat sat In", None, 4), ("The cat sat In In", None, 5)],
    )
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    await _settle(ws)
    ws.put({"type": "video.query", "text": "q"})
    await _settle(ws)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=5.0)

    types = ws.sent_types()
    assert types.count("response.start") == 1
    assert types.count("response.text.done") == 1
    deltas = [m["delta"] for m in ws.sent if m.get("type") == "response.text.delta"]
    assert "".join(deltas) == "The cat sat"
    assert not any("In" in d for d in deltas)


@pytest.mark.asyncio
async def test_sink_frames_zero_warns_opening_recall_not_preserved(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        video_stream_persistent.logger,
        "warning",
        lambda message, *args, **_kwargs: warnings.append(message % args if args else message),
    )

    engine = FakeEngineClient(["ok"])
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg(sink_frames=0))
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=5.0)

    assert any("opening recall not preserved" in w for w in warnings)
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_sink_frames_set_does_not_warn(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        video_stream_persistent.logger,
        "warning",
        lambda message, *args, **_kwargs: warnings.append(message % args if args else message),
    )

    engine = FakeEngineClient(["ok"])
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg(sink_frames=1))
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=5.0)

    assert not any("opening recall not preserved" in w for w in warnings)


@pytest.mark.asyncio
async def test_audio_modality_warns_text_only(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        video_stream_persistent.logger,
        "warning",
        lambda message, *args, **_kwargs: warnings.append(message % args if args else message),
    )

    engine = FakeEngineClient(["ok"])
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg(modalities=["text", "audio"]))
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=5.0)

    assert any("text-output only" in w for w in warnings)
    assert engine.calls == [] or engine.calls[0]["output_modalities"] == ["text"]


@pytest.mark.asyncio
async def test_engine_failure_sends_error_and_ends_session():
    engine = FakeEngineClient(fail=True)
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})

    await asyncio.wait_for(task, timeout=5.0)

    assert {"type": "error", "message": "Persistent session epoch failed"} in ws.sent
    assert "session.done" in ws.sent_types()
    assert engine.epochs == 1  # the epoch loop breaks instead of retrying


@pytest.mark.asyncio
async def test_persistent_session_without_engine_client_sends_error():
    ws = MockWebSocket([json.dumps(_config_msg())])
    handler = QwenOmniStreamingVideoHandler(chat_service=object(), engine_client=None)

    await handler.handle_session(ws)

    assert {"type": "error", "message": "Streaming video requires an engine client"} in ws.sent
    assert "session.done" not in [m.get("type") for m in ws.sent]


@pytest.mark.asyncio
async def test_query_before_any_frame_reports_no_frames_buffered():
    engine = FakeEngineClient(["ok"])
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "too early"})
    await _settle(ws)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    await _settle(ws)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=5.0)

    assert {"type": "error", "message": "No frames buffered"} in ws.sent
    assert engine.chunks[0]["text"] == _SEED_PREFIX + _VID_BLOCK
    assert "session.done" in ws.sent_types()


class HangingEngineClient:
    """Engine that drains the input stream but never ends the output stream.

    Mirrors the S2 session-close hang: the terminal output for the final
    ``resumable=False`` sentinel never reaches the client, so ``generate()``'s
    async-for never finishes.
    """

    def __init__(self) -> None:
        self.drained = asyncio.Event()
        self.epochs = 0

    def generate(self, *, prompt, sampling_params=None, request_id="", output_modalities=None):
        self.epochs += 1

        async def _run():
            async def _drain() -> None:
                async for _chunk in prompt:
                    pass
                self.drained.set()

            task = asyncio.create_task(_drain())
            try:
                await asyncio.sleep(3600)  # terminal output never arrives
                yield  # pragma: no cover - unreachable, keeps _run an async generator
            finally:
                task.cancel()

        return _run()


@pytest.mark.asyncio
async def test_stream_end_guard_closes_session_when_engine_never_terminates(monkeypatch, caplog):
    """A dropped engine terminal must degrade to a warning + `session.done`."""
    monkeypatch.setattr(video_stream_persistent, "_PERSIST_STREAM_END_TIMEOUT", 0.2)
    engine = HangingEngineClient()
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    await _settle(ws)
    with caplog.at_level("WARNING"):
        ws.put({"type": "video.done"})
        await asyncio.wait_for(task, timeout=5.0)

    assert engine.drained.is_set()  # the epoch generator did complete
    assert "session.done" in ws.sent_types()
    assert "error" not in ws.sent_types()
    assert any("engine stream did not end" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_stream_end_guard_does_not_fire_on_a_clean_close(caplog):
    """The guard is belt-and-braces: a healthy engine closes before it can fire."""
    engine = FakeEngineClient(["ok"])
    ws = TimedWebSocket()
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    await _settle(ws)
    with caplog.at_level("WARNING"):
        ws.put({"type": "video.done"})
        await asyncio.wait_for(task, timeout=5.0)

    assert "session.done" in ws.sent_types()
    assert not any("engine stream did not end" in r.getMessage() for r in caplog.records)


class DisconnectingWebSocket(TimedWebSocket):
    """Client that vanishes mid-answer: the socket dies after N deltas.

    Once dead every send raises, which is what starlette does after the peer
    closes the connection.
    """

    def __init__(self, fail_after_deltas: int = 1):
        super().__init__()
        self._fail_after = fail_after_deltas
        self._deltas = 0
        self.dead = False

    async def send_json(self, data: dict[str, Any]):
        if self.dead:
            raise WebSocketDisconnect(code=1001)
        if data.get("type") == "response.text.delta":
            self._deltas += 1
            if self._deltas >= self._fail_after:
                self.dead = True
                raise WebSocketDisconnect(code=1001)
        await super().send_json(data)


@pytest.mark.asyncio
async def test_client_disconnect_mid_answer_closes_session_quietly(caplog):
    """A dead browser ends the session: no error frame, no epoch-failed
    traceback, and the engine request is torn down."""
    engine = FakeEngineClient(["The", "cat", "sat"])
    ws = DisconnectingWebSocket(fail_after_deltas=1)
    handler = _persistent_handler(engine)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    await _settle(ws)
    with caplog.at_level("INFO"):
        ws.put({"type": "video.query", "text": "what is happening?"})
        await asyncio.wait_for(task, timeout=5.0)

    assert ws.dead
    types = ws.sent_types()
    assert "error" not in types  # a disconnect is not an error to report
    assert "session.done" not in types  # the guarded send failed silently
    assert engine.aborted  # result stream closed -> engine request released
    assert engine.epochs == 1  # no retry loop
    assert any("client disconnected" in r.getMessage() for r in caplog.records)
    assert not any("epoch failed" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_idle_timeout_mid_epoch_closes_session_cleanly(caplog):
    """The reader's idle timeout takes the normal done path: one error frame
    naming the timeout, then a clean close."""
    engine = FakeEngineClient(["ok"])
    ws = TimedWebSocket()
    handler = QwenOmniStreamingVideoHandler(chat_service=object(), engine_client=engine, idle_timeout=0.2)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(_config_msg())
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10))})
    await _settle(ws)
    with caplog.at_level("WARNING"):
        # No further client traffic: the reader's idle timeout fires mid-epoch.
        await asyncio.wait_for(task, timeout=5.0)

    assert {"type": "error", "message": "Idle timeout"} in ws.sent
    assert "session.done" in ws.sent_types()
    assert engine.epochs == 1
    assert not any("epoch failed" in r.getMessage().lower() for r in caplog.records)
    assert not any("engine stream did not end" in r.getMessage() for r in caplog.records)
