#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demo client for the persistent streaming-video session (Qwen3-Omni / vLLM-Omni).

Streams a video clip frame-by-frame to a running omni server over WebSocket and
asks a two-part question periodically:
  (a) what is happening right now, and (b) what was shown at the very beginning.

It prints each answer and a final PASS/checks summary, so you can confirm the four
always-online signals on your own hardware:
  1. current-moment (a) tracks the evolving video,
  2. opening recall (b) stays consistent across refreshes,
  3. (server log) KV `alive` stays flat — see the test guide,
  4. the session ends cleanly with no errors.

Usage:
  python demo_client.py --video clip.mp4 --port 8901 \
      --frames 800 --query-every 50 --refresh-at 2000 --sink 6 --recent 10

Requires: `websockets`, `pillow`, and a frame reader (`vllm.assets.video` if vLLM is
installed, otherwise `opencv-python`). The server must already be serving (see the
test guide / run_server.sh).
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import urllib.request

from PIL import Image

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

QUERY = (
    "In one sentence each: (a) what is happening right now, and (b) what was shown at the very BEGINNING of the video?"
)


def read_frames(path: str, n: int):
    """Return a list of [H,W,3] uint8 RGB frames sampled from the video."""
    try:
        from vllm.assets.video import video_to_ndarrays

        arr = video_to_ndarrays(path, num_frames=n)
        return [arr[i] for i in range(arr.shape[0])]
    except Exception:
        import numpy as np  # noqa: F401

        try:
            import cv2
        except ImportError:
            sys.exit("Install a frame reader: vLLM (vllm.assets.video) or opencv-python")
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
        step = max(1, total // n)
        frames, i = [], 0
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(f[:, :, ::-1].copy())  # BGR->RGB
            i += 1
        cap.release()
        return frames


def jpeg_b64(nd) -> str:
    buf = io.BytesIO()
    Image.fromarray(nd).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def model_id(port: int) -> str:
    with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


async def main(a) -> int:
    frames = read_frames(a.video, a.frames)
    n = len(frames)
    model = model_id(a.port)
    uri = f"ws://{a.host}:{a.port}/v1/video/chat/stream"
    n_q = sum(1 for i in range(1, n + 1) if i % a.query_every == 0)
    print(
        f"[demo] model={model} frames={n} queries~={n_q} refresh_at={a.refresh_at} sink={a.sink} recent={a.recent}",
        flush=True,
    )

    answers, errors, got_done = [], [], False

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        config = {
            "type": "session.config",
            "model": model,
            "modalities": ["text"],
            "persistent": True,
            "sink_frames": a.sink,
            "num_frames": a.recent,
            "refresh_at_position": a.refresh_at,
            "enable_frame_filter": False,
        }
        if a.brief:
            config["system_prompt"] = (
                "You are a video understanding assistant. Answer in exactly two short "
                "sentences: one for (a) and one for (b). Do not add any other text."
            )
        await ws.send(json.dumps(config))

        async def sender():
            for i in range(n):
                # Encode off the event loop: a synchronous encode per frame
                # starves websocket keepalive pongs under flood (1011 close).
                data = await asyncio.to_thread(jpeg_b64, frames[i])
                await ws.send(json.dumps({"type": "video.frame", "data": data}))
                if (i + 1) % a.query_every == 0:
                    await ws.send(json.dumps({"type": "video.query", "text": QUERY}))
                    await asyncio.sleep(0.2)
            await ws.send(json.dumps({"type": "video.done"}))

        async def receiver():
            nonlocal got_done
            cur = ""
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=600)
                d = json.loads(raw)
                t = d.get("type")
                if t == "response.start":
                    cur = ""
                elif t == "response.text.delta":
                    cur += d.get("delta", "")
                elif t == "response.text.done":
                    final = (d.get("text", "") or cur).strip()
                    answers.append(final)
                    b = final.split("(b)", 1)[1].strip()[:220] if "(b)" in final else "(missing)"
                    a_ = final.split("(b)", 1)[0].strip()[:220]
                    print(f"\n[Q{len(answers)}] (a) {a_}\n      (b) {b}", flush=True)
                elif t == "error":
                    errors.append(d.get("message", ""))
                    print(f"[ERROR] {d.get('message')}", flush=True)
                elif t == "session.done":
                    got_done = True
                    return

        send_task = asyncio.create_task(sender())
        try:
            await receiver()
        finally:
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)

    print("\n===== SUMMARY =====", flush=True)
    print(f"answers={len(answers)}  errors={len(errors)}  session_done={got_done}", flush=True)
    ok = got_done and not errors and len(answers) >= 1 and all(x for x in answers)
    print(f"checks: clean close={got_done}  no errors={not errors}  all answered={ok}", flush=True)
    print(
        "Now confirm in the SERVER log: `[streaming-kv] eviction ... alive=N` stays flat and "
        "new `vsess-...-<epoch>` ids appear (one per refresh).",
        flush=True,
    )
    print(f"RESULT: {'PASS' if ok else 'CHECK LOG'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="path to a video file")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--frames", type=int, default=800, help="frames to stream (crosses refreshes)")
    p.add_argument("--query-every", type=int, default=50, help="frames between questions")
    p.add_argument("--refresh-at", type=int, default=2000, help="M-RoPE position estimate to refresh at (>=1024)")
    p.add_argument("--sink", type=int, default=6, help="opening frames pinned + re-seeded each epoch")
    p.add_argument("--recent", type=int, default=10, help="recent frames re-seeded for continuity")
    p.add_argument("--brief", action="store_true", help="send a brevity system prompt (cleaner answers)")
    sys.exit(asyncio.run(main(p.parse_args())))
