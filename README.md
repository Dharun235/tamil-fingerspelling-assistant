# Tamil Fingerspelling Hand Pose

CPU hand-pose extraction for still images. This repository contains only the hand-pose pipeline; Tamil-character classification comes later.

## Pipeline

```text
image → BlazePalm detector → hand crop → 21 landmarks per hand
```

For each detected hand, runtime output includes:

- 21 `(x, y, z)` landmarks
- hand-presence confidence
- handedness probability
- annotated skeleton image

For classifier training, `build_landmark_dataset.py` saves normalized landmark coordinates in a CSV. Image pixels are not copied into that CSV.

No MediaPipe Python runtime, GUI, or GPU required. Inference uses ONNX Runtime CPU.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download these weights into `weights/`:

- `palm_detection_full_Nx3x192x192.onnx`
- `hand_landmark_full_Nx3x224x224.onnx`

Use download links in the [upstream model README](https://github.com/yakhyo/mediapipe-hand-landmark-onnx#models). `weights/`, `data/`, and `output/` are Git-ignored.

## Run

```bash
python scripts/hand_pose_onnx.py \
  "data/TLFS23 - Tamil Language Finger Spelling Image Dataset/Dataset Folders/1/img_001.jpg" \
  --save-dir output
```

Multiple images:

```bash
python scripts/hand_pose_onnx.py \
  "data/TLFS23 - Tamil Language Finger Spelling Image Dataset/Dataset Folders/1/img_001.jpg" \
  "data/TLFS23 - Tamil Language Finger Spelling Image Dataset/Dataset Folders/2/img_001.jpg" \
  --save-dir output
```

Output image shows hand skeleton, landmarks, handedness, and confidence.

## Build landmark dataset

Dataset folders become class labels. For example, every image under `Dataset Folders/1/` gets label `1`.

```bash
python scripts/build_landmark_dataset.py \
  "data/TLFS23 - Tamil Language Finger Spelling Image Dataset/Dataset Folders" \
  --output-csv data/landmarks.csv
```

For a faster balanced experiment, process 100 sampled images per class:

```bash
python scripts/build_landmark_dataset.py \
  "data/TLFS23 - Tamil Language Finger Spelling Image Dataset/Dataset Folders" \
  --max-per-class 100 \
  --output-csv data/landmarks_100.csv
```

CSV contains one row per image, two fixed hand slots (`left`, `right`), confidence values, and 21 normalized `(x, y, z)` landmarks per hand. Coordinates are wrist-relative and scale-normalized, so image resolution and hand position matter less. `data/landmarks.csv` is ignored by Git.

## Landmark order

```text
0       wrist
1–4     thumb
5–8     index
9–12    middle
13–16   ring
17–20   pinky
```

The model returns `(N, 21, 3)` landmarks for `N` hands. Later MLP training can use two fixed hand slots, wrist-relative normalization, hand scale, confidence, and hand-center position.

## Prepare, train, evaluate, predict

Create a stratified 80/20 train/test split. No validation split used:

```bash
python scripts/prepare_dataset.py data/landmarks.csv
```

Train and save MLP:

```bash
python scripts/train_mlp.py \
  --train data/splits/train.npz \
  --model output/mlp.joblib
```

Evaluate held-out test data:

```bash
python scripts/evaluate.py \
  --model output/mlp.joblib \
  --test data/splits/test.npz
```

Save up to 100 correct and 100 wrong test images for inspection:

```bash
python scripts/save_failures.py \
  --model output/mlp.joblib \
  --test data/splits/test.npz
```

Outputs are separated into `output/cases/correct/` and `output/cases/wrong/`, with class folders. Each case stores `original.jpg`, annotated `prediction.jpg`, side-by-side `comparison.jpg`, and an `index.csv` manifest.

Run live webcam prediction:

```bash
python scripts/realtime_predict.py --model output/mlp.joblib
```

Press `q` or `Esc` to quit. The classifier predicts only classes included during training.

## Dataset attribution

Example images come from **TLFS23 – Tamil Language Finger Spelling Image Dataset**, containing 248 classes and 255,155 images. Dataset license: **CC BY 4.0**. Credit:

> Chirranjeavi M, Bavesh Ram S, Gokulraj Varatharajan, Aaruran Sundaresh, Binoy Nair, Harikumar M E. *TLFS23 - Tamil Language Finger Spelling Image Dataset*. Mendeley Data, 2023. DOI: [10.17632/39kzs5pxmk.2](https://doi.org/10.17632/39kzs5pxmk.2).

Dataset page: [Mendeley Data](https://data.mendeley.com/datasets/39kzs5pxmk/2). Research article: [TLFS23 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10790027/).

## Model attribution

ONNX inference implementation and converted hand models adapted from [yakhyo/mediapipe-hand-landmark-onnx](https://github.com/yakhyo/mediapipe-hand-landmark-onnx). Upstream repository identifies the implementation and weights as Apache-2.0. Original hand model topology comes from [Google MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).

## Files

```text
scripts/hand_pose_onnx.py  # inference CLI
scripts/build_landmark_dataset.py  # images → landmark CSV
scripts/prepare_dataset.py  # CSV → train/test arrays
scripts/train_mlp.py        # train split → saved classifier
scripts/evaluate.py         # test split → metrics/confusion matrix
scripts/save_failures.py    # save misclassified test images
scripts/realtime_predict.py # webcam → landmarks → class text
models/                    # ONNX model architecture/post-processing
requirements.txt           # CPU dependencies
```
