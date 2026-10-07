"""Local browser server for Tamil Fingerspelling Assistant."""

from __future__ import annotations

import json
import io
import os
import re
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.pipeline import RealtimePipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data/TLFS23 - Tamil Language Finger Spelling Image Dataset"
DATA_ROOT = Path(os.environ.get("TLFS_DATA_ROOT", DEFAULT_DATA_ROOT))
REFERENCE_DIR = Path(os.environ.get("TLFS_REFERENCE_DIR", DATA_ROOT / "Refrence Image"))
LABELS_PATH = Path(os.environ.get("TLFS_LABELS_PATH", DATA_ROOT / "ReadMe.txt"))
app = FastAPI(title="Tamil Fingerspelling Assistant")
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")
if REFERENCE_DIR.exists():
    app.mount("/references", StaticFiles(directory=REFERENCE_DIR), name="references")


@app.on_event("startup")
async def startup_message():
    if not cv2.__version__.startswith("5."):
        raise RuntimeError(f"OpenCV 5 is required; found {cv2.__version__}")
    print(f"[app] READY: OpenCV {cv2.__version__}", flush=True)
    print("[app] Open http://127.0.0.1:8000", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/references")
def references():
    """Return Tamil-character to reference-image mappings for reverse display."""
    if not REFERENCE_DIR.exists():
        return {"available": False, "references": {}}
    labels = {}
    if not LABELS_PATH.exists():
        return {"available": False, "references": {}}
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if parts and parts[0].isdigit() and len(parts) > 1:
            labels[parts[0]] = parts[1].split("(", 1)[0].strip()
    result = {}
    for image in REFERENCE_DIR.iterdir():
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = re.match(r"^(\d+)[_-]", image.name)
        if not match:
            continue
        number = match.group(1)
        character = labels.get(number)
        if character and character != "Background":
            result[character] = {"class_id": number, "url": f"/references/{quote(image.name)}"}
    return {"available": bool(result), "references": result}


@app.websocket("/ws")
async def websocket(websocket: WebSocket):
    await websocket.accept()
    print("[app] CLIENT CONNECTED: loading RTMPose", flush=True)
    # RTMLib prints its own low-level loading lines. Keep terminal output focused
    # on this application and replace them with one clear status message.
    with redirect_stdout(io.StringIO()):
        pipeline = RealtimePipeline()
    print("[app] MODEL READY: RTMPose + ONNXRuntime CPU", flush=True)
    inference_started = False
    try:
        while True:
            message = await websocket.receive()
            if message.get("text"):
                command = json.loads(message["text"])
                pipeline.command(command.get("action", ""))
                continue
            data = message.get("bytes")
            if not data:
                continue
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "invalid frame"})
                continue
            # OpenCV 5 normalizes frames before RTMPose inference.
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            result = pipeline.process(frame)
            if not inference_started:
                print("[app] INFERENCE ACTIVE", flush=True)
                inference_started = True
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("[app] CLIENT DISCONNECTED", flush=True)
        return
    except Exception as error:
        # Some Starlette/Uvicorn versions raise a differently named disconnect
        # exception after a browser closes the socket during inference.
        if error.__class__.__name__ in {"WebSocketDisconnected", "WebSocketDisconnect"}:
            print("[app] CLIENT DISCONNECTED", flush=True)
            return
        raise
