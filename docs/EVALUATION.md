# Evaluation

Run the reproducible benchmark across all 247 Tamil classes:

```bash
python scripts/evaluate_dataset.py
```

This evaluates 4,940 images: 20 images from each of the 247 non-background classes.

```text
results/dataset_eval_20/
├── predictions.csv  # one row per image, stage outputs and latency
├── per_class.csv    # class support and exact class accuracy
└── summary.json     # hand, finger, class, and latency metrics
```

Run the complete dataset only when needed:

```bash
python scripts/evaluate_dataset.py --limit-per-class 0
```

The full TLFS23 dataset contains roughly 255k images and can take hours on CPU. Use `--limit-per-class 0` only when a complete sweep is needed. Use the summary's `latency_ms` values to report inference performance. This is image inference latency only; it excludes browser camera capture, JPEG transfer, WebSocket scheduling, and display rendering.

## Meaning of the metrics

- **Hand stage:** whether the expected left/right hand set from the folder mapping was detected.
- **Finger stage:** binary open/closed precision, recall, F1, and accuracy for each finger, conditional on a hand being detected.
- **Class stage:** whether the observed hand-state combination maps exactly to the folder class.
- **Latency:** RTMPose detection, duplicate suppression, finger rules, and mapping per image.

The folder mapping is a proxy label. It does not prove that every photographed hand pose is manually correct. Report missing/overlapping-hand failures separately from rule-based finger errors.

The static benchmark and live experience answer different questions. The static benchmark processes every selected image without user correction. The live interface is human-in-the-loop: the signer sees the hand/finger overlay, improves framing when needed, and then holds the sign for commitment. Manual testing showed reliable behavior for the tested sign combinations under clear framing, but this observation is qualitative until a fixed live protocol is recorded.

## Latest sample run

Command:

```bash
python scripts/evaluate_dataset.py
```

The latest run evaluated 20 images from each of the 247 non-background classes: 4,940 images on CPU.

| Stage | Result |
|---|---:|
| Expected left hands detected | 84.5% (3,956/4,680) |
| Expected right hands detected | 83.5% (3,826/4,580) |
| Exact detected hand-side set | 70.7% |
| Finger-state accuracy | 92.5–97.3% per finger |
| Exact folder-class match, full set | 54.8% |
| Mean image latency | 116.6 ms |
| p95 image latency | 193.3 ms |
| Mean image throughput | 8.6 images/s |

Outputs are in `results/dataset_eval_20/`:

```text
predictions.csv          # every image and all intermediate stages
per_class.csv            # support and class accuracy for classes 1–247
summary.json             # aggregate metrics and latency
```

Interpretation: conditional finger rules are substantially stronger than the final class score suggests. The main bottleneck in this photographed dataset is missing/merged/occluded hand detection and landmark quality, not only the on/off rule. The class result is not publication-grade ground-truth accuracy because folder labels are proxy labels.


## Real-time evaluation

Offline image latency is not end-to-end live latency. For the live demo, record a short fixed protocol instead of all 247 signs:

1. Run 10–20 representative signs, including one-hand, two-hand, fist/space, overlap, and difficult framing.
2. Hold each sign until the 350 ms stability bar commits it; repeat each sign 3 times.
3. Record committed text, missed commits, duplicate commits, and visible delay.
4. Report median/p95 commit latency separately from server inference latency.

The browser-to-server path adds camera capture, JPEG encoding, network/WebSocket transfer, rendering, and the stability timer. A small scripted live protocol captures those effects more honestly than treating every frame as an independent classification.
