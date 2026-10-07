# Architecture

The source diagram is [`architecture.mmd`](architecture.mmd). GitHub can render the Mermaid source, or it can be exported to SVG/PNG for Devpost.

## Runtime path

1. Browser captures JPEG frames.
2. FastAPI receives frames through a WebSocket.
3. OpenCV 5 decodes and normalizes each frame to 640×480.
4. RTMPose/RTMDet predicts up to two hands and landmarks using ONNXRuntime CPU.
5. Geometry rules classify each finger as straight or bent.
6. Tamil mapping converts the combination into a Tamil character.
7. Stability timing prevents repeated letters from one held pose; a fist inserts a space.
8. Browser renders Tamil text and green/red feedback.

## Processing stages

The system is inference-only; there is no project-specific training stage.

| Stage | Input | Output | Saved during evaluation |
|---|---|---|---|
| 1. Decode | Camera JPEG/image | OpenCV BGR frame | image path, latency |
| 2. Hand detection | BGR frame | Up to two hand keypoint arrays | detected hand count, confidence |
| 3. Landmark cleanup | Keypoints | Duplicate-suppressed left/right hands | expected/detected hand sides |
| 4. Finger state | 21 landmarks/hand | T/I/M/R/P straight or bent states | expected/observed states, per-finger correctness |
| 5. Tamil mapping | One/two hand states | Tamil character or unknown | predicted class, class correctness |
| 6. Temporal commit | Character stream | Stable character, space, or no commit | live commit event and delay |

The offline evaluator writes one CSV row per image so every stage can be inspected independently. It does not hide hand-stage failures inside the finger-rule score.

## Offline and live paths

The dataset folders are a replaceable evaluation fixture. Folder labels provide expected hand combinations and finger states. The evaluator accepts `--dataset-root`, processes a fixed number of images per class, writes stage outputs, and computes hand, finger, class, and latency metrics. This is a proxy because the folder mapping is not manually verified landmark ground truth.

The browser captures frames and sends JPEGs through a WebSocket. The server runs the same detector and rules used by the offline evaluator. The browser renders intermediate feedback before the stability timer commits text. The signer can reposition a poorly framed hand; this is an intended human-in-the-loop property.

## AWS path

The Docker image is pushed to private ECR. ECS Express Mode runs it as an x86-64 Fargate service behind an HTTPS Application Load Balancer. CloudWatch receives container logs. IAM roles allow ECS to pull the image, write logs, and manage Express Mode resources.

The browser camera remains client-side. Frames are not intentionally persisted.
