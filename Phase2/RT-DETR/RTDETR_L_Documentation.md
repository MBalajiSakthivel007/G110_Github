# RT-DETR-L — Model Documentation

**Project:** Coastal Human Surveillance: Dual-Zone Pedestrian & Swimmer Risk Analytics via High-Resolution Vision Detectors  

**Model owner:** Chitturi Venkata Chaitanya (CB.SC.U4CSE23214)  

**Guide:** Ms. R. Sujee, Assistant Professor, Department of CSE, Amrita Vishwa Vidyapeetham, Coimbatore

---

## 1. Overview

This model detects every person in a frame from a fixed beach camera and classifies each one as **on land (class 0)** or **in the water (class 1)**. From these detections it produces per-frame land / water head-counts.

It is a **real-time detection transformer**: attention lets it see the whole frame, and its one-to-one set prediction needs **no NMS** — useful for dense groups of swimmers. It is trained, selected and evaluated with exactly the same data and rules as the other four models, so all results can be compared directly.

| Item | Value |
|---|---|
| Framework | Ultralytics 8.4 (PyTorch) |
| Base model | `rtdetr-l.pt` (COCO-pretrained) |
| Model family | real-time detection transformer |
| Task | 2-class object detection: person on land / person in water |
| Dataset | [G110-BEACH-PEOPLE-FINAL](https://app.roboflow.com/mbalajis-workspace-g2gae/g110-beach-people-final/2), Roboflow version 2 |

---

## 2. How the model works

![RT-DETR-L architecture](../Architecture/05_RTDETR_L_Architecture.png)

*Architecture of the model as implemented in our notebook: the detection pipeline (top), the training setup and the test results (bottom).*

Step by step:

```
Input frame (1920x1080, letterboxed to 1280 px)
   -> HGNetv2 CNN backbone           : multi-scale features (rtdetr-l COCO weights)
   -> Hybrid encoder                 : AIFI - self-attention on the deepest map
                                       CCFM - convolutional fusion across scales
   -> IoU-aware query selection      : the best 300 encoder features become object queries
   -> Transformer decoder            : refines the queries into boxes and classes
   -> Set prediction                 : one box per object (Hungarian 1-to-1 matching) - NO NMS
   -> Post-processing                : score >= 0.60, boxes shorter than 15 px removed
   -> Output                         : land and water boxes + per-frame land / water counts
```

- **No NMS:** each object gets exactly one prediction, so neighbouring swimmers are not merged and no time is spent on NMS.
- **Loss:** varifocal loss for classes, L1 + GIoU for boxes.

---

## 3. Dataset

The dataset was built by the team for this project from two public, fixed beach webcams.

| Split | Frames | Boxes | Land | Water |
|---|---|---|---|---|
| Train | 3,454 | 132,027 | 57,634 | 74,393 |
| Validation | 782 | 31,443 | 13,558 | 17,885 |
| Test (people ≥ 15 px) | 399 | 11,918 | 4,546 | 7,372 |

- **Cameras:** White Bay, British Virgin Islands (crowded) and Sunny Isles, USA (far, small people).
- **Frames:** one frame every 3 s from 48 clips of about 5 minutes, 1920×1080; stream overlays masked.
- **Labels:** draft boxes → YOLOv8x labeller → verified cleaning rules; land / water decided by the person's foot point relative to the waterline of each camera.
- **Split:** by clip, so frames of one clip never appear in two splits; the test clips come from a different day.
- **Test set:** completed by eye; people under 15 px are an *ignore region* — detecting them is neither rewarded nor punished.

---

## 4. Training configuration

| Setting | Value |
|---|---|
| Input size | 1280 px (letterboxed) |
| Training length | fixed time budget of 8.5 h — Ultralytics fits epochs and learning-rate schedule to the time |
| GPUs | 2 × Tesla T4 (DDP) |
| Early stopping | patience 15 epochs |
| Augmentation | Ultralytics defaults |
| Epochs reached | 70 in 8.5 h |
| Checkpoint selection | best epoch on validation (`best.pt`) |
| Score threshold | chosen on validation → **0.60** |

---

## 5. Step-by-step: what was done

1. **Download the dataset** — G110-BEACH-PEOPLE-FINAL v2 from Roboflow.
2. **Mark small people** — test boxes shorter than 15 px become ignore regions (same rule for all 5 models).
3. Load the COCO `rtdetr-l.pt` model.
4. Train on 2 GPUs within an 8.5 h budget (70 epochs); Ultralytics keeps the best epoch as `best.pt`.
5. **Choose the score threshold** — predict on the validation split and pick the threshold that maximises F1(land) + F1(water); the test set is never used for this choice.
6. **Evaluate once on the test split** — COCO mAP, per-class precision / recall / F1, 3×3 confusion matrix, results per camera and per crowd level, per-frame counting error and speed.
7. **With / without study** — repeat the test prediction with denoising, CLAHE, different NMS thresholds, test-time augmentation and the plain COCO model.
8. **Render outputs** — ground-truth vs model sample frames and output videos.

---

## 6. Results (test split: 399 frames, 4,546 land + 7,372 water people)

### 6.1 Main metrics

| Metric | Value |
|---|---|
| **mAP@50** | **80.94 %** (best of 5) |
| mAP@50:95 | **44.83 %** (best of 5) |
| AP50 land / water | 81.65 % / **80.23 %** (best water) |
| Precision / recall — land | 76.31 % / 76.33 % |
| Precision / recall — water | 81.59 % / 77.93 % |
| F1 (support-weighted) | 78.42 % |
| Speed | **4.64 frames/s** at 1280 px (fastest of 5) |

### 6.2 Per camera and counting error

| | White Bay (crowded) | Sunny Isles (far, small people) |
|---|---|---|
| mAP@50 | **83.78 %** (best of 5) | 68.96 % |
| Count error, land (people / frame) | 2.11 | 1.84 |
| Count error, water (people / frame) | **2.57** | 1.53 |

### 6.3 With / without study (AP50, %)

| Variant | AP50 |
|---|---|
| **Baseline (raw frames)** | **80.94** (best variant) |
| Bilateral denoising | 79.02 |
| CLAHE | 79.80 |
| Denoising + CLAHE | 78.15 |
| NMS 0.3 / 0.7 | 80.94 / 80.94 * |
| Test-time augmentation | 80.94 * |
| COCO model, no fine-tuning | 66.55 |

### 6.4 Reading the results

- **Best model overall:** highest mAP, best on swimmers and the fastest — the strongest candidate for a live system.
- Land↔water mix-ups are the lowest of all models (0.07 % / 0.42 %); it has the highest recall (about 95 % land, 88 % water).
- \* RT-DETR has no NMS and its Ultralytics version does not run test-time augmentation, so these variants equal the baseline.

---

## 7. Figures

| Training curves | Confusion matrix (row-normalised) |
|---|---|
| ![Training curves](Output/Charts/01_training_curves.png) | ![Confusion matrix](Output/Charts/05_confusion_full_3x3_normalised.png) |

| Precision–recall curves | Results per camera and crowd level |
|---|---|
| ![PR curves](Output/Charts/06_pr_curves.png) | ![Per scenario](Output/Charts/08_scenario_results.png) |

**Ground truth (left) vs model (right)** — green = land, blue = water

![White Bay sample](Output/Sample_Frames/12_sample_whitebay_1_GT_vs_RT-DETR.jpg)
![Sunny Isles sample](Output/Sample_Frames/12_sample_sunnyisles_1_GT_vs_RT-DETR.jpg)

---

## 8. Output videos

All videos are in [`Output/Videos`](Output/Videos) (H.264 MP4). Click a file, then *View raw* to play it.

| Video | Shows |
|---|---|

| `video_GT_vs_RT-DETR_ALL_test_clips.mp4` | ground truth vs model on all four test clips |

| `video_GT_vs_RT-DETR_sunnyisles_001_20261110_224944.mp4` | ground truth vs model, test clip Sunny Isles 001 |

| `video_GT_vs_RT-DETR_sunnyisles_002_20261110_225435.mp4` | ground truth vs model, test clip Sunny Isles 002 |

| `video_GT_vs_RT-DETR_whitebay_001_20261110_224945.mp4` | ground truth vs model, test clip White Bay 001 |

| `video_GT_vs_RT-DETR_whitebay_002_20261110_225438.mp4` | ground truth vs model, test clip White Bay 002 |

| `video_continuous_sunnyisles_001_20261110_224944.mp4` | model at real speed with live land / water counts, Sunny Isles clip 001 |

| `video_continuous_sunnyisles_002_20261110_225435.mp4` | model at real speed with live land / water counts, Sunny Isles clip 002 |

| `video_continuous_whitebay_001_20261110_224945.mp4` | model at real speed with live land / water counts, White Bay clip 001 |

| `video_continuous_whitebay_002_20261110_225438.mp4` | model at real speed with live land / water counts, White Bay clip 002 |

---

## 9. How to reproduce on Kaggle

1. Create a Kaggle notebook with **GPU T4** and **Internet** enabled.
2. Upload the notebook from [`Notebooks/`](Notebooks).
3. Add your Roboflow API key as a Kaggle secret named `ROBOFLOW_API_KEY` (Add-ons → Secrets), or paste it where the notebook says `PASTE_YOUR_ROBOFLOW_KEY_HERE`. **Never commit a real key.**
4. Run all cells — the notebook downloads the dataset, trains, keeps the best checkpoint and writes all results and videos.

5. To skip training, download `RTDETR_L_best.pt` (and `RTDETR_L_last.pt`) from the [repository releases](https://github.com/MBalajiSakthivel007/G110_Github/releases) and load it in the evaluation part of the notebook.

---

## 10. Files in this folder

| Path | Content |
|---|---|

| `Notebooks/RTDETR_L_v2.ipynb` | Kaggle notebook with all saved outputs |

| `Output/Charts/` | training curves, confusion matrices, precision–recall curves, threshold sweep, per-camera results, counts over time |
| `Output/Sample_Frames/` | ground truth vs model frames and pre-processing comparisons |
| `Output/Metrics/` | all numbers: COCO metrics, confusion, per camera, counts, threshold choice, configuration used |

| `Output/Training_Run/` | Ultralytics training run: loss curves, `results.csv`, training / validation batches, arguments used |

| `Output/Videos/` | 9 output videos (see section 8) |

All results are also in the [OneDrive results folder](https://amritavishwavidyapeetham-my.sharepoint.com/:f:/g/personal/cb_sc_u4cse23213_cb_students_amrita_edu/IgDjuQtqn8jnRoQ1-mjkN0uEAc-A2ImPLlxZCtwB1lXY6T8?e=fdj88C).

---

## 11. Limitations

- Two cameras, daylight only; very small or fully hidden people are still missed.

- Transformers converge slowly; validation mAP@50 was flat after about 40 epochs but mAP@50:95 was still rising slowly.

- Training length was limited by Kaggle's 12-hour session limit.

## 12. References (IEEE)

1. Y. Zhao *et al.*, "DETRs Beat YOLOs on Real-time Object Detection," in *Proc. IEEE/CVF CVPR*, 2024, pp. 16965–16974.
2. S. Ma *et al.*, "OWRT-DETR: A Novel Real-Time Transformer Network for Small-Object Detection in Open-Water Search and Rescue From UAV Aerial Imagery," *IEEE TGRS*, vol. 63, 2025.
