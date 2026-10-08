#!/usr/bin/env python3
"""Measure warm WebSocket round-trip latency against the live service."""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import websockets


async def measure(endpoint: str, image_path: Path, count: int = 10):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"could not read {image_path}")
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("could not encode frame")
    payload = encoded.tobytes()
    async with websockets.connect(endpoint, max_size=4 * 1024 * 1024) as socket:
        await socket.send(payload)
        await socket.recv()
        samples = []
        for _ in range(count):
            started = time.perf_counter()
            await socket.send(payload)
            response = json.loads(await socket.recv())
            samples.append((time.perf_counter() - started) * 1000)
            if "error" in response:
                raise RuntimeError(response["error"])
        print(json.dumps({
            "frames": count,
            "mean_ms": round(statistics.mean(samples), 2),
            "p50_ms": round(statistics.median(samples), 2),
            "p95_ms": round(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)], 2),
            "samples_ms": [round(value, 2) for value in samples],
        }, indent=2))


if __name__ == "__main__":
    url = sys.argv[1]
    image = Path(sys.argv[2])
    asyncio.run(measure(url, image))
