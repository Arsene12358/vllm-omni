"""The omni schedulers must drain the base scheduler's deferred error sets.

``OmniARScheduler.update_from_output`` and
``OmniGenerationScheduler.update_from_output`` are full re-implementations that
never call ``super()``. The base ``Scheduler`` defers two classes of
per-request failure to ``update_from_output`` -- grammar compilation failures
and streaming sessions whose next input chunk would cross ``max_model_len`` --
by recording the request id and letting that method finish it with
``FinishReason.ERROR``. If the omni overrides do not drain those sets, the
request is never finished and never emits an output, so its client waits
forever: on the persistent video session that turns the max_model_len wall from
a clean per-request error back into a silent hang.

Behaviour is pinned on the shared helper; placement (that both overrides
actually call it) is pinned separately, because driving a full
``update_from_output`` needs a model runner output and a live KV cache manager.
"""

from __future__ import annotations

import inspect
from collections import defaultdict

import pytest
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SCHEDULER_PARAMS = [
    pytest.param(OmniARScheduler, id="ar"),
    pytest.param(OmniGenerationScheduler, id="generation"),
]


class _StubRequest:
    def __init__(self, request_id: str, stop_reason: str | None = None) -> None:
        self.request_id = request_id
        self.client_index = 0
        self.stop_reason = stop_reason
        self.trace_headers = None

    def get_finished_reason(self) -> FinishReason:
        return FinishReason.ERROR

    def take_events(self):
        return None


def _make_scheduler(scheduler_cls, *, overflow=(), grammar=()):
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.streaming_overflow_error_reqs = set(overflow)
    scheduler.grammar_compile_error_reqs = set(grammar)
    return scheduler


@pytest.mark.parametrize("scheduler_cls", _SCHEDULER_PARAMS)
def test_drain_finishes_overflowed_sessions_with_an_error(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls, overflow=["session"])
    finished: list[set[str]] = []
    request = _StubRequest("session", stop_reason="... max_model_len of 4096 ...")

    def _finish_requests(req_ids, status):
        finished.append((set(req_ids), status))
        return [request]

    scheduler.finish_requests = _finish_requests
    outputs: dict[int, list] = defaultdict(list)

    scheduler._drain_deferred_error_reqs(outputs)

    assert finished == [({"session"}, RequestStatus.FINISHED_ERROR)]
    emitted = outputs[0]
    assert len(emitted) == 1
    assert emitted[0].request_id == "session"
    assert emitted[0].finish_reason == FinishReason.ERROR
    assert "max_model_len" in emitted[0].stop_reason  # the reason reaches the client
    assert not scheduler.streaming_overflow_error_reqs  # drained, so it fires once


@pytest.mark.parametrize("scheduler_cls", _SCHEDULER_PARAMS)
def test_drain_covers_grammar_failures_too(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls, overflow=["session"], grammar=["grammar"])
    seen: set[str] = set()

    def _finish_requests(req_ids, status):
        seen.update(req_ids)
        return [_StubRequest(r) for r in req_ids]

    scheduler.finish_requests = _finish_requests
    scheduler._drain_deferred_error_reqs(defaultdict(list))

    assert seen == {"session", "grammar"}
    assert not scheduler.grammar_compile_error_reqs


@pytest.mark.parametrize("scheduler_cls", _SCHEDULER_PARAMS)
def test_drain_is_a_noop_when_nothing_is_pending(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls)

    def _finish_requests(req_ids, status):  # pragma: no cover - must not run
        raise AssertionError("finish_requests called with nothing pending")

    scheduler.finish_requests = _finish_requests
    outputs: dict[int, list] = defaultdict(list)
    scheduler._drain_deferred_error_reqs(outputs)
    assert not outputs


@pytest.mark.parametrize("scheduler_cls", _SCHEDULER_PARAMS)
def test_drain_tolerates_schedulers_without_the_sets(scheduler_cls):
    """__new__-constructed schedulers (used across these tests) lack the sets."""
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler._drain_deferred_error_reqs(defaultdict(list))


@pytest.mark.parametrize("scheduler_cls", _SCHEDULER_PARAMS)
def test_update_from_output_drains_deferred_errors(scheduler_cls):
    """Placement: the override must actually call the drain, or the sets are
    never emptied and the session hangs."""
    source = inspect.getsource(scheduler_cls.update_from_output)
    assert "_drain_deferred_error_reqs" in source
