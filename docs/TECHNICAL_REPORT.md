# Technical report draft

## Problem

Tamil fingerspelling users need immediate feedback while producing individual signs. The assistant must show what it understood, avoid registering a held sign repeatedly, support spaces, and display the result in Tamil script.

## Solution

The application detects up to two hands, extracts 2D landmarks, classifies individual finger states from normalized geometry, maps the state combination to TLFS23 fingerspelling labels, and commits a character only after a stable hold. Green/red overlays let the signer correct framing before committing. In the reverse direction, typed Tamil text selects the corresponding TLFS23 reference image, giving the user a visual hand-pose target.

This is an inference-and-integration system rather than a model-training system. It combines pretrained hand-pose models, deterministic geometric rules, a fixed fingerspelling table, and a temporal user-interface state machine.

## OpenCV 5 implementation

OpenCV 5 performs JPEG decoding and fixed-size frame normalization before the RTMPose/RTMDet hand-pose pipeline. OpenCV is pinned to `5.0.0.93` and checked at application startup.

## AWS implementation

The container runs on ECS Express Mode using Linux x86-64 Fargate in Stockholm (`eu-north-1`). Express Mode provides the Fargate service, public HTTPS Application Load Balancer, autoscaling, networking, and monitoring. ECR stores the image and CloudWatch stores container logs.

## Evaluation plan

Report results separately for:

1. Hand detection and landmark completeness.
2. Per-finger straight/bent classification.
3. End-to-end Tamil character recognition.
4. Runtime latency and throughput on the deployed service.

Use reference images and held-out TLFS23 images. Include per-class metrics, latency percentiles, and representative failure images. Do not hide missed or merged hands inside the final character score.

The main dataset run uses 20 images from each of the 247 non-background classes (4,940 images). It writes `predictions.csv` with every image and intermediate stage output, `per_class.csv`, and `summary.json` with aggregate metrics and latency percentiles under `results/dataset_eval_20/`.

The live interface is intentionally human-in-the-loop. The user receives immediate pose feedback, adjusts hand placement when the camera view is poor, and holds the corrected sign until the stability timer commits it. Manual testing showed reliable behavior for the tested combinations under clear framing; this is reported as a qualitative observation rather than an unsupported statistical accuracy claim.

Real-time evaluation uses a short fixed protocol: repeated one-hand, two-hand, overlap, fist/space, and difficult-framing signs. It measures committed-character accuracy, missed/duplicate commits, median/p95 commit delay, and user-visible behavior. End-to-end latency includes capture, JPEG encoding, network transfer, inference, rendering, and the 350 ms stability requirement.

## Limitations and responsible use

This is an assistive prototype, not a certified accessibility or language-translation device. Occlusion, out-of-frame hands, overlapping hands, lighting, camera angle, and landmark errors can change the result. The interface exposes intermediate feedback so users can decide whether to repeat a sign. No frame is intentionally persisted.
