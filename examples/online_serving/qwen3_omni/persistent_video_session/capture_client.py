# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture client for the demo recording: streams frames + periodic queries and
prints timestamped OUT events (answers) so a side monitor's GPU-memory / KV-alive
samples can be aligned into one timeline and rendered into a short video.

Env: PORT, VIDEO_PATH, N_FRAMES, QUERY_EVERY, REFRESH_AT, SINK_FRAMES, NUM_FRAMES.
"""

import asyncio
import base64
import io
import json
import os
import sys
import time
import urllib.request

import websockets
from PIL import Image
from vllm.assets.video import video_to_ndarrays

PORT = int(os.environ.get("PORT", "8901"))
VIDEO = os.environ.get("VIDEO_PATH", "sample.mp4")
N_FRAMES = int(os.environ.get("N_FRAMES", "500"))
QUERY_EVERY = int(os.environ.get("QUERY_EVERY", "80"))
REFRESH_AT = int(os.environ.get("REFRESH_AT", "2000"))
SINK = int(os.environ.get("SINK_FRAMES", "6"))
RECENT = int(os.environ.get("NUM_FRAMES", "10"))
QUERY = "In one sentence each: (a) what is happening right now, and (b) what was shown at the very beginning?"
BRIEF = (
    "You are a video understanding assistant. Answer in exactly two short sentences: "
    "one for (a) and one for (b). Do not add any other text."
)


def jpeg_b64(nd) -> str:
    buf = io.BytesIO()
    Image.fromarray(nd).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def model_id() -> str:
    with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def out(ev, **kw):
    print(f"OUT t={time.time():.3f} ev={ev} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


async def main() -> int:
    frames = video_to_ndarrays(VIDEO, num_frames=N_FRAMES)
    n = int(frames.shape[0])
    model = model_id()
    uri = f"ws://localhost:{PORT}/v1/video/chat/stream"
    out("config", frames=n, query_every=QUERY_EVERY, refresh_at=REFRESH_AT)

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "model": model,
                    "modalities": ["text"],
                    "persistent": True,
                    "sink_frames": SINK,
                    "num_frames": RECENT,
                    "refresh_at_position": REFRESH_AT,
                    "enable_frame_filter": False,
                    "system_prompt": BRIEF,
                }
            )
        )

        done = asyncio.Event()

        async def receiver():
            cur = ""
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=600)
                except asyncio.TimeoutError:
                    return
                d = json.loads(raw)
                t = d.get("type")
                if t == "response.start":
                    cur = ""
                elif t == "response.text.delta":
                    cur += d.get("delta", "")
                elif t == "response.text.done":
                    final = (d.get("text", "") or cur).strip()
                    out("answer", text=json.dumps(final))
                elif t == "session.done":
                    out("session_done")
                    done.set()
                    return

        recv = asyncio.create_task(receiver())
        out("stream_start", frames=n)
        for i in range(n):
            # Encode off the event loop: a synchronous encode per frame starves
            # websocket keepalive pongs under flood (1011 close).
            data = await asyncio.to_thread(jpeg_b64, frames[i])
            await ws.send(json.dumps({"type": "video.frame", "data": data}))
            if (i + 1) % 25 == 0:
                out("sent", n=i + 1)
            if (i + 1) % QUERY_EVERY == 0:
                out("query", n=i + 1)
                await ws.send(json.dumps({"type": "video.query", "text": QUERY}))
                await asyncio.sleep(0.2)
        await ws.send(json.dumps({"type": "video.done"}))
        try:
            await asyncio.wait_for(done.wait(), timeout=120)
        finally:
            recv.cancel()
            await asyncio.gather(recv, return_exceptions=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
