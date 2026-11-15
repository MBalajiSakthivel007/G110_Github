# Faster R-CNN R50-FPN — Model Documentation

**Project:** Coastal Human Surveillance: Dual-Zone Pedestrian & Swimmer Risk Analytics via High-Resolution Vision Detectors  

**Model owner:** Bhargava Ram Kaza (CB.SC.U4CSE23711)  

**Guide:** Ms. R. Sujee, Assistant Professor, Department of CSE, Amrita Vishwa Vidyapeetham, Coimbatore

---

## 1. Overview

This model detects every person in a frame from a fixed beach camera and classifies each one as **on land (class 0)** or **in the water (class 1)**. From these detections it produces per-frame land / water head-counts.

It uses exactly the same pipeline as Faster R-CNN R101-FPN with a **lighter ResNet-50 backbone**, so the pair measures what backbone depth buys in accuracy and speed. It is trained, selected and evaluated with exactly the same data and rules as the other four models, so all results can be compared directly.

| Item | Value |
|---|---|
| Framework | Detectron2 0.6 (PyTorch) |
| Base model | `COCO-Detection/faster_rcnn_R_50_FPN_3x` (COCO-pretrained) |
| Model family | two-stage detector with a lighter backbone |
| Task | 2-class object detection: person on land / person in water |
| Dataset | [G110-BEACH-PEOPLE-FINAL](https://app.roboflow.com/mbalajis-workspace-g2gae/g110-beach-people-final/2), Roboflow version 2 |

---

## 2. How the model works

![Faster R-CNN R50-FPN architecture](../Architecture/03_FasterRCNN_R50_Architecture.png)

*Architecture of the model as implemented in our notebook: the detection pipeline (top), the training setup and the test results (bottom).*

Step by step:

```
Input frame (1920x1080, resized to 1320 px)
   -> ResNet-50 backbone             : residual CNN (50 layers), extracts visual features
   -> Feature Pyramid Network (FPN)  : feature maps P2-P6, one per object size
   -> Region Proposal Network (RPN)  : stage 1 - anchors of 8, 16, 32, 64, 128 px x 3 aspect ratios
   -> RoIAlign (7x7) + box head      : stage 2 - land / water / background + box refinement
   -> Post-processing                : NMS (IoU 0.5), score >= 0.60, boxes shorter than 15 px removed
   -> Output                         : land and water boxes + per-frame land / water counts
```

- **Backbone-depth study:** everything except the backbone is identical to the R101 model — same data, anchors, schedule and evaluation.
- **Loss:** RPN and RoI head classification (cross-entropy) and box regression (smooth-L1), as in the R101 model.

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
| Pretrained weights | COCO (Detectron2 model zoo) |
| Iterations | 8,000 × batch 2 images ≈ 4.6 epochs |
| Learning rate | 0.0025, 500 warm-up iterations, ×0.1 at 70 % and 90 % of training |
| Mixed precision | enabled (AMP) |
| Training sizes | shortest side 1080 / 1200 / 1320 px (max 2400) |
| Test size | 1320 px (max 2400) |
| Anchors | 8, 16, 32, 64, 128 px × 3 aspect ratios |
| RoI batch per image | 256 |
| Checkpoint selection | validation AP50 every 500 iterations; best checkpoint kept |
| Score threshold | chosen on validation → **0.60** |
| Hardware | 1 × Tesla T4 (16 GB) |

---

## 5. Step-by-step: what was done

1. **Download the dataset** — G110-BEACH-PEOPLE-FINAL v2 from Roboflow.
2. **Mark small people** — test boxes shorter than 15 px become ignore regions (same rule for all 5 models).
3. **Configure Detectron2** — load the R50-FPN COCO model, set 2 classes, small anchors and high resolution.
4. Train for 8,000 iterations; keep the checkpoint with the best validation AP50.
5. **Choose the score threshold** — predict on the validation split and pick the threshold that maximises F1(land) + F1(water); the test set is never used for this choice.
6. **Evaluate once on the test split** — COCO mAP, per-class precision / recall / F1, 3×3 confusion matrix, results per camera and per crowd level, per-frame counting error and speed.
7. **With / without study** — repeat the test prediction with denoising, CLAHE, different NMS thresholds, test-time augmentation and the plain COCO model.
8. **Render outputs** — ground-truth vs model sample frames and output videos.

---

## 6. Results (test split: 399 frames, 4,546 land + 7,372 water people)

### 6.1 Main metrics

| Metric | Value |
|---|---|
| **mAP@50** | **77.50 %** |
| mAP@50:95 | 40.15 % |
| AP50 land / water | 82.01 % / 73.00 % |
| Precision / recall — land | 80.33 % / 78.42 % |
| Precision / recall — water | 79.64 % / 76.29 % |
| F1 (support-weighted) | 78.47 % |
| Speed | 1.70 frames/s at 1320 px (30 % faster than R101) |

### 6.2 Per camera and counting error

| | White Bay (crowded) | Sunny Isles (far, small people) |
|---|---|---|
| mAP@50 | 80.84 % | 66.84 % |
| Count error, land (people / frame) | 1.96 | 1.36 |
| Count error, water (people / frame) | 2.88 | 1.45 |

### 6.3 With / without study (AP50, %)

| Variant | AP50 |
|---|---|
| Baseline (raw frames) | 77.50 |
| Bilateral denoising | 74.70 |
| **CLAHE** | **79.47** (best variant) |
| Denoising + CLAHE | 76.60 |
| NMS 0.3 / 0.7 | 75.94 / 77.89 |
| Test-time augmentation | 77.81 |
| COCO model, no fine-tuning | not run |

### 6.4 Reading the results

- Land↔water mix-ups stay below 1 %; slightly more people are missed than with R101.
- Compared with R101 it loses only **1.74 mAP@50 points** but runs **30 % faster** — a cheaper Faster R-CNN.
- **CLAHE** gives the largest gain of all models (+2.0) by reducing glare.

---

## 7. Figures

| Training curves | Confusion matrix (row-normalised) |
|---|---|
| ![Training curves](Output/Charts/01_training_curves.png) | ![Confusion matrix](Output/Charts/05_confusion_full_3x3_normalised.png) |

| Precision–recall curves | Results per camera and crowd level |
|---|---|
| ![PR curves](Output/Charts/06_pr_curves.png) | ![Per scenario](Output/Charts/08_scenario_results.png) |

**Ground truth (left) vs model (right)** — green = land, blue = water

![White Bay sample](Output/Sample_Frames/12_sample_whitebay_1_GT_vs_FasterRCNN.jpg)
![Sunny Isles sample](Output/Sample_Frames/12_sample_sunnyisles_1_GT_vs_FasterRCNN.jpg)

---

## 8. Output videos

All videos are in [`Output/Videos`](Output/Videos) (H.264 MP4). Click a file, then *View raw* to play it.

| Video | Shows |
|---|---|

| `video_GT_vs_FasterRCNN_ALL_test_clips.mp4` | ground truth vs model on all four test clips |

| `video_GT_vs_FasterRCNN_sunnyisles_001_20261110_224944.mp4` | ground truth vs model, test clip Sunny Isles 001 |

| `video_GT_vs_FasterRCNN_sunnyisles_002_20261110_225435.mp4` | ground truth vs model, test clip Sunny Isles 002 |

| `video_GT_vs_FasterRCNN_whitebay_001_20261110_224945.mp4` | ground truth vs model, test clip White Bay 001 |

| `video_GT_vs_FasterRCNN_whitebay_002_20261110_225438.mp4` | ground truth vs model, test clip White Bay 002 |

---

## 9. How to reproduce on Kaggle

1. Create a Kaggle notebook with **GPU T4** and **Internet** enabled.
2. Upload the notebook from [`Notebooks/`](Notebooks).
3. Add your Roboflow API key as a Kaggle secret named `ROBOFLOW_API_KEY` (Add-ons → Secrets), or paste it where the notebook says `PASTE_YOUR_ROBOFLOW_KEY_HERE`. **Never commit a real key.**
4. Run all cells — the notebook downloads the dataset, trains, keeps the best checkpoint and writes all results and videos.

---

## 10. Files in this folder

| Path | Content |
|---|---|

| `Notebooks/FasterRCNN_R50_v2.ipynb` | Kaggle notebook with all saved outputs |

| `Output/Charts/` | training curves, confusion matrices, precision–recall curves, threshold sweep, per-camera results, counts over time |
| `Output/Sample_Frames/` | ground truth vs model frames and pre-processing comparisons |
| `Output/Metrics/` | all numbers: COCO metrics, confusion, per camera, counts, threshold choice, configuration used |

| `Output/Videos/` | 5 output videos (see section 8) |

All results are also in the [OneDrive results folder](https://amritavishwavidyapeetham-my.sharepoint.com/:f:/g/personal/cb_sc_u4cse23213_cb_students_amrita_edu/IgCBYt1e0mKfSZhmW2fN3xDJARs1ZIUn-w1qTMYSKtTrqag?e=Ed1qLf).

---

## 11. Limitations

- Two cameras, daylight only; very small or fully hidden people are still missed.

- This run produced the ground-truth vs model videos only (no real-speed videos, because the test-video input was not attached to the notebook).

- Training length was limited by Kaggle's 12-hour session limit.

## 12. References (IEEE)

1. S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," *IEEE TPAMI*, vol. 39, no. 6, pp. 1137–1149, 2017.
2. O. Hmidani and E. M. Ismaili Alaoui, "A comprehensive survey of the R-CNN family for object detection," in *Proc. IEEE CommNet*, 2022, pp. 1–6.
3. X. Qian, Q. Jian, W. Wang, X. Yao, and G. Cheng, "Incorporating Multiscale Context and Task-Consistent Focal Loss into Oriented Object Detection," *IEEE TGRS*, vol. 63, 2025.
4. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.
