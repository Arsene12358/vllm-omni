#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the captured demo segment into a short MP4 (+GIF).

Parses MON (GPU mem / KV-alive / tokens) and OUT (answers) lines from a capture
job log and draws a dashboard per output frame: tokens processed climbing while
the live KV working set and GPU memory stay flat, with model answers appearing.

Usage: python render_demo.py CAPTURE_LOG  [--out demo.mp4] [--fps 15] [--speed 1.0]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

# warm-paper palette (matches the HTML showcase)
IVORY = (250, 249, 245)
PAPER = (255, 255, 255)
OAT = (227, 218, 204)
SLATE = (20, 20, 19)
CLAY = (217, 119, 87)
CLAY_D = (184, 92, 62)
OLIVE = (120, 140, 93)
G300 = (209, 207, 197)
G500 = (135, 134, 127)
G700 = (61, 61, 58)
W, H = 1280, 720


def font(sz, bold=False):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf" % ("-Bold" if bold else "-Regular"),
    ]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, sz)
    return ImageFont.load_default()


def parse(path):
    mon, events = [], []

    def kv(s, k):
        return (re.search(rf"{k}=([0-9.]+)", s) or [None, None])[1]

    for line in open(path, errors="ignore"):
        if "MON t=" in line:
            t = kv(line, "t")
            mem = kv(line, "mem")
            comp = kv(line, "computed")
            alive = kv(line, "alive")
            if t and mem:
                mon.append(
                    {
                        "t": float(t),
                        "mem": float(mem),
                        "computed": float(comp) if comp else None,
                        "alive": float(alive) if alive else None,
                    }
                )
        elif "OUT t=" in line:
            t = kv(line, "t")
            m = re.search(r"ev=(\w+)", line)
            if not (t and m):
                continue
            ev = {"t": float(t), "ev": m.group(1)}
            if ev["ev"] == "answer":
                am = re.search(r"text=(.*)$", line.rstrip())
                try:
                    ev["text"] = json.loads(am.group(1)) if am else ""
                except Exception:
                    ev["text"] = am.group(1) if am else ""
            sm = re.search(r"\bn=([0-9]+)", line)
            if sm:
                ev["n"] = int(sm.group(1))
            events.append(ev)
    return mon, events


def cumulative_tokens(mon):
    """computed resets to ~0 at each refresh; accumulate across resets."""
    offset = 0.0
    prev = 0.0
    out = []
    for s in mon:
        c = s["computed"]
        if c is None:
            out.append(offset + prev)
            continue
        if c + 1 < prev:  # reset (refresh)
            offset += prev
            s["refresh"] = True
        prev = c
        out.append(offset + c)
    return out


def wrap(draw, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def rrect(d, box, r, fill, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def draw_frame(state, fonts):
    img = Image.new("RGB", (W, H), IVORY)
    d = ImageDraw.Draw(img)
    f_h1, f_h2, f_big, f_lbl, f_body, f_mono = fonts
    # header
    d.text((40, 28), "Always-Online Streaming Video — live session", font=f_h1, fill=SLATE)
    d.text((40, 66), "Qwen3-Omni · bounded KV · opening retained across refreshes", font=f_lbl, fill=G700)
    clock = f"{int(state['t']) // 60:02d}:{int(state['t']) % 60:02d}"
    d.text((W - 150, 30), clock, font=f_h2, fill=CLAY_D)

    # metric cards
    cards = [
        ("TOKENS PROCESSED", f"{int(state['tokens']):,}", "stream growing", CLAY_D, False),
        ("LIVE KV BLOCKS", f"{int(state['alive'])}", "BOUNDED", OLIVE, True),
        ("GPU MEMORY (GPU0)", f"{state['mem'] / 1024:.1f} GB", "CONSTANT", OLIVE, True),
    ]
    cw, gap, x0, y0 = 380, 20, 40, 104
    for i, (lbl, val, tag, col, good) in enumerate(cards):
        x = x0 + i * (cw + gap)
        rrect(d, [x, y0, x + cw, y0 + 120], 12, PAPER, OAT, 1)
        d.text((x + 18, y0 + 14), lbl, font=f_lbl, fill=G700)
        d.text((x + 18, y0 + 40), val, font=f_big, fill=SLATE)
        rrect(
            d,
            [x + 18, y0 + 92, x + 18 + int(d.textlength(tag, font=f_lbl)) + 18, y0 + 114],
            10,
            (235, 240, 230) if good else (250, 238, 232),
        )
        d.text((x + 27, y0 + 95), tag, font=f_lbl, fill=col)

    # chart: tokens (rising) vs KV alive (flat)
    cx0, cy0, cx1, cy1 = 40, 268, W - 40, 500
    rrect(d, [cx0, cy0, cx1, cy1], 12, PAPER, OAT, 1)
    d.text((cx0 + 18, cy0 + 12), "Tokens processed vs. live KV blocks", font=f_lbl, fill=G700)
    px0, py0, px1, py1 = cx0 + 60, cy0 + 44, cx1 - 24, cy1 - 30
    d.line([px0, py1, px1, py1], fill=G300, width=1)
    d.line([px0, py0, px0, py1], fill=G300, width=1)
    span = max(1e-6, state["span"])
    series_t = state["series_t"]
    series_tok = state["series_tok"]
    series_alive = state["series_alive"]
    tok_max = max(state["tok_max"], 1)
    alive_ref = state.get("alive_ref", 673)

    def X(t):
        return px0 + (t / span) * (px1 - px0)

    # tokens line (clay, rising)
    pts = [(X(t), py1 - (v / tok_max) * (py1 - py0)) for t, v in zip(series_t, series_tok) if t <= state["t"]]
    if len(pts) > 1:
        d.line(pts, fill=CLAY, width=3, joint="curve")
    # KV alive line (olive, flat) — scaled against a generous max so it reads as flat-low
    apts = [
        (X(t), py1 - (a / (alive_ref * 4)) * (py1 - py0))
        for t, a in zip(series_t, series_alive)
        if t <= state["t"] and a is not None
    ]
    if len(apts) > 1:
        d.line(apts, fill=OLIVE, width=3)
    # refresh markers
    for rt in state["refresh_ts"]:
        if rt <= state["t"]:
            x = X(rt)
            d.line([x, py0, x, py1], fill=(217, 119, 87), width=1)
    d.text((px0, py0 - 2), "tokens", font=f_lbl, fill=CLAY_D)
    if apts:
        d.text((apts[-1][0] - 80, apts[-1][1] - 22), "KV blocks (flat)", font=f_lbl, fill=OLIVE)
    d.text((px1 - 150, py1 + 8), "↑ rises   |   ↓ stays flat", font=f_lbl, fill=G500)

    # model output box
    oy0 = 516
    rrect(d, [40, oy0, W - 40, H - 30], 12, (251, 247, 242), OAT, 1)
    d.text(
        (58, oy0 + 12),
        "MODEL OUTPUT" + ("   ● answering…" if state["answering"] else ""),
        font=f_lbl,
        fill=CLAY_D if state["answering"] else G700,
    )
    ans = state["answer"] or "(streaming video… ask a question)"
    lines = wrap(d, ans, f_body, W - 130)[:5]
    for i, ln in enumerate(lines):
        d.text((58, oy0 + 40 + i * 26), ln, font=f_body, fill=SLATE)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()

    mon, events = parse(a.log)
    if not mon:
        sys.exit("no MON samples found in log")
    toks = cumulative_tokens(mon)
    t0 = mon[0]["t"]
    for s, tk in zip(mon, toks):
        s["tt"] = s["t"] - t0
        s["tok"] = tk
    span = mon[-1]["tt"]
    refresh_ts = [s["tt"] for s in mon if s.get("refresh")]
    answers = [(e["t"] - t0, e.get("text", "")) for e in events if e["ev"] == "answer"]
    queries = [e["t"] - t0 for e in events if e["ev"] == "query"]
    series_t = [s["tt"] for s in mon]
    series_tok = [s["tok"] for s in mon]
    series_alive = [s["alive"] for s in mon]
    tok_max = max(series_tok) if series_tok else 1
    alive_vals = [a for a in series_alive if a]
    alive_ref = max(alive_vals) if alive_vals else 673

    fonts = (font(30, True), font(30, True), font(40, True), font(15), font(20), font(16))
    tmp = tempfile.mkdtemp(prefix="demo_frames_")
    n_out = int(span * a.fps / a.speed) + 1
    print(
        f"span={span:.1f}s  samples={len(mon)}  answers={len(answers)}  refreshes={len(refresh_ts)}  "
        f"tok_max={int(tok_max):,}  alive_ref={int(alive_ref)}  -> {n_out} frames"
    )

    def sample_at(tt):
        s = mon[0]
        for m in mon:
            if m["tt"] <= tt:
                s = m
            else:
                break
        return s

    for k in range(n_out):
        tt = k * a.speed / a.fps
        s = sample_at(tt)
        ans = ""
        for at, txt in answers:
            if at <= tt:
                ans = txt
        answering = any(at <= tt < at + 1.6 for at in queries) and not any(
            qt <= aat <= tt for qt in queries for aat, _ in answers if qt <= aat
        )
        answering = any(qt <= tt < qt + 2.5 and not any(qt <= at2 <= tt for at2, _ in answers) for qt in queries)
        st = {
            "t": tt,
            "span": span,
            "tokens": s["tok"],
            "alive": s["alive"] or alive_ref,
            "mem": s["mem"],
            "answer": ans,
            "answering": answering,
            "series_t": series_t,
            "series_tok": series_tok,
            "series_alive": series_alive,
            "tok_max": tok_max,
            "alive_ref": alive_ref,
            "refresh_ts": refresh_ts,
        }
        draw_frame(st, fonts).save(os.path.join(tmp, f"f{k:05d}.png"))

    mp4 = a.out
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(a.fps),
            "-i",
            os.path.join(tmp, "f%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("wrote", mp4)
    if a.gif:
        pal = os.path.join(tmp, "pal.png")
        gif = mp4.rsplit(".", 1)[0] + ".gif"
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp4, "-vf", "fps=10,scale=900:-1:flags=lanczos,palettegen", pal],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                mp4,
                "-i",
                pal,
                "-lavfi",
                "fps=10,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse",
                gif,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("wrote", gif)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
