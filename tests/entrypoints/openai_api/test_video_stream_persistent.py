# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the persistent streaming-LLM video session building blocks.

Covers the ``_TextStreamDemux`` event demultiplexer and the persistent-session
fields on ``StreamingVideoSessionConfig``. The persistent session driver itself
is exercised separately.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.video_stream_base import StreamingVideoSessionConfig
from vllm_omni.entrypoints.openai.video_stream_persistent import _TextStreamDemux

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
