# Five-minute demo script

## 0:00–0:30 — Need

Introduce the accessibility goal: convert individual Tamil fingerspelling signs into visible Tamil text with immediate feedback.

## 0:30–1:20 — Live interaction

Open the endpoint. Allow camera access. Show a sign and hold it until the progress bar commits. Point out Tamil output, preview, and green/red finger overlays.

## 1:20–2:00 — Continuous input

Show several signs in sequence. Demonstrate one character per held sign, a new timer for a new sign, and a fist producing a space. Press `Stop camera`, then `Enable camera`.

## 2:00–2:30 — Reverse reference

Type a Tamil character in the reference box. Show the matching TLFS23 hand image and explain that it gives the signer a visual target for the reverse direction.

## 2:30–3:20 — Architecture

Show the diagram: browser, WebSocket, OpenCV 5, RTMPose/RTMDet, geometric rules, Tamil mapping, ECR, ECS/Fargate, ALB, and CloudWatch.

## 3:20–4:10 — Evidence

Show the repository, pinned dependencies, `/health` response, and Docker/AWS instructions. Show representative successful and failed hand detections.

## 4:10–4:40 — Limitations

Demonstrate an occluded or overlapping-hand case. Explain that the overlay makes the failure visible and lets the signer reposition hands before committing.

## 4:40–5:00 — Impact

Close with the goal: low-friction, real-time fingerspelling assistance through a browser and reproducible AWS container deployment.
