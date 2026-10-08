# Tamil Fingerspelling Assistant

Real-time Tamil fingerspelling assistance from a webcam. The browser displays detected finger state, gives immediate visual feedback, and converts stable signs into Tamil Unicode text.

## Pipeline

```text
webcam → browser JPEG → FastAPI/WebSocket → OpenCV 5 preprocessing
        → RTMPose hand detection + landmarks → geometric finger rules
        → Tamil mapping → Tamil text
```

The system is frame-based. A sign commits after 350 ms of stability. A fist commits a space. The user can stop/start the camera without stopping the AWS service.

This is an inference-and-integration project, not a model-training project. It combines pretrained RTMPose/RTMDet, geometric finger-state rules, Tamil sign mapping, temporal stability, and a browser/AWS runtime.

## Demo

Live endpoint:

```text
https://ta-89507d6158f8458c88ac7b38194d8efc.ecs.eu-north-1.on.aws
```

The endpoint is enabled for demonstrations only and may be offline between sessions.

Local browser demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and allow camera access. Frames are not written to disk.

The reverse reference preview uses a replaceable dataset asset root. The default TLFS23 layout is:

```text
data/TLFS23 - Tamil Language Finger Spelling Image Dataset/
├── ReadMe.txt
└── Refrence Image/
```

The dataset is intentionally excluded from Git. Obtain it separately from the dataset owner before building Docker. The camera-only pipeline can still run without the reference images; the reverse preview then reports that references are unavailable.

To use another compatible dataset, preserve this layout and set the paths before starting the server:

```bash
export TLFS_DATA_ROOT=/path/to/dataset
export TLFS_REFERENCE_DIR=/path/to/dataset/Refrence\ Image
export TLFS_LABELS_PATH=/path/to/dataset/ReadMe.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

The reference filenames must begin with a numeric class ID, for example `34-Ki.jpg`, and `ReadMe.txt` must map that ID to the displayed Tamil character. The offline evaluator accepts a replacement dataset directly:

```bash
python scripts/evaluate_dataset.py --dataset-root /path/to/dataset
```

### TLFS23 dataset attribution

This project uses the **TLFS23 - Tamil Language Finger Spelling Image Dataset** for Tamil labels, reference images, and offline evaluation. Obtain the dataset from the original [Mendeley Data record](https://data.mendeley.com/datasets/39kzs5pxmk/2), DOI [10.17632/39kzs5pxmk.2](https://doi.org/10.17632/39kzs5pxmk.2), and comply with its **CC BY 4.0** license. Cite: Chirranjeavi, M., Bavesh Ram, S., Gokulraj Varatharajan, Aaruran Sundaresh, Binoy B. Nair, and Harikumar M. E., “TLFS23 - Tamil Language Finger Spelling Image Dataset,” Mendeley Data, version 2, 2023. The dataset is excluded from Git and is not redistributed by this project.

## Docker

```bash
docker compose up --build
```

Open `http://127.0.0.1:8000`. The container uses ONNXRuntime CPU inference and downloads RTMPose weights on first use. Compose persists the model cache.

For a replacement reference dataset in Docker, mount it read-only and set the same variables:

```bash
docker run --rm -p 8000:8000 \
  -v /path/to/dataset:/app/custom_dataset:ro \
  -e TLFS_DATA_ROOT=/app/custom_dataset \
  -e TLFS_REFERENCE_DIR="/app/custom_dataset/Refrence Image" \
  -e TLFS_LABELS_PATH=/app/custom_dataset/ReadMe.txt \
  tamil-fingerspelling-fingerspelling:latest
```

## AWS deployment

The service runs on ECS Express Mode in `eu-north-1` (Stockholm): Linux x86-64 Fargate, an HTTPS Application Load Balancer, autoscaling, and CloudWatch logs. The Docker image is stored in private ECR.

Build and push from Apple Silicon:

```bash
docker buildx build \
  --platform linux/amd64 \
  -t 108464427575.dkr.ecr.eu-north-1.amazonaws.com/tamil-fingerspelling:v5 \
  --push .
```

Start or stop the public service:

```bash
./aws_start.sh
./aws_stop.sh
```

Stopping removes the Express runtime infrastructure. Starting recreates it from ECR and may produce a new public URL.

The current AWS image tag is `v5`.

## User interaction

- Green overlay: straight/open finger.
- Red overlay: bent/closed finger.
- Center text: committed Tamil message.
- Preview: current sign.
- Progress bar: stability time before commit.
- `Stop camera` releases webcam access and stops frame uploads.
- A fist inserts a space after the stability timer completes.
- The reverse preview accepts typed Tamil text and displays the matching reference sign image from TLFS23.

## Repository layout

```text
core/pipeline.py                 Frame-to-text state machine
models/rtmpose_utils.py          Hand post-processing
models/geometry_features.py      Normalized landmark geometry
models/finger_rules.py            Finger open/closed rules
models/tamil_labels.py            TLFS23 Tamil labels
scripts/realtime.py               Desktop webcam demo
server.py                         FastAPI/WebSocket server
web/                              Browser interface
aws_start.sh / aws_stop.sh        AWS lifecycle scripts
docs/                             Submission report, mapping, diagrams, and deck
```

## Evaluation and limitations

This project has two evaluation modes:

1. **Static dataset evaluation:** 20 images from each of 247 non-background classes, 4,940 images total. This measures detector, finger-state, Tamil mapping, and image latency under photographed dataset conditions.
2. **Interactive real-time evaluation:** the signer sees the detected hand pose and green/red finger feedback, adjusts hand placement, and holds the sign until commitment. In manual testing, the tested sign combinations worked reliably when the hands were clearly visible and correctly framed. This is a qualitative human-in-the-loop observation, not a formal live accuracy percentage.

The static dataset contains more out-of-frame, occluded, merged, and poorly framed images than the interactive use case. Therefore its 54.8% exact folder-class result should not be presented as the real-time user experience, and the live observation should not be presented as a statistically measured accuracy. Report both conditions separately. No model training is required by the current pipeline.

Run the reproducible dataset benchmark:

```bash
python scripts/evaluate_dataset.py
```

See the evaluation and latency tables in [`docs/Tamil_Fingerspelling_Technical_Report.pdf`](docs/Tamil_Fingerspelling_Technical_Report.pdf). The raw benchmark outputs remain under `results/dataset_eval_20/`.

Known limitations:

- RTMPose can miss or merge hands when they overlap or leave the frame.
- Rules are sensitive to occlusion and unusual camera angles.
- Real-time performance benefits from the visible feedback loop: users can reposition their hands before a sign is committed.
- This is fingerspelling assistance, not continuous Tamil sign-language translation.
- AWS demo uses CPU inference; latency depends on network and Fargate load.

## License

Project code is MIT licensed. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset images and model weights retain their own licenses and are not redistributed here.

## Submission artifacts

- [`docs/Tamil_Fingerspelling_Technical_Report.pdf`](docs/Tamil_Fingerspelling_Technical_Report.pdf) — formatted technical report.
- [`docs/TLFS23_CLASS_MAPPING.md`](docs/TLFS23_CLASS_MAPPING.md) — complete 247-class mapping.
- [`docs/Tamil_Fingerspelling_Demo_Deck.pptx`](docs/Tamil_Fingerspelling_Demo_Deck.pptx) — presentation deck.
- `docs/assets/` — report and presentation diagrams.
