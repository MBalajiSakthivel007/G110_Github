<div align="center">

# 🏖️ Coastal Human Surveillance

### Dual-Zone Pedestrian & Swimmer Risk Analytics via High-Resolution Vision Detectors

*Find every person on a beach camera, tell who is **on the sand** and who is **in the water**, and count both — frame by frame.*

[![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Detectron2](https://img.shields.io/badge/Detectron2-0.6-4B8BBE)](https://github.com/facebookresearch/detectron2)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4-111F68)](https://github.com/ultralytics/ultralytics)
[![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle%20Tesla%20T4-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[🌐 Website](https://mbalajisakthivel007.github.io/G110_Website/)** ·
**[📘 Documentation](docs/Phase2_Project_Documentation.md)** ·
**[📄 Report](Phase2/Report/G110_Phase2_Report.pdf)** ·
**[📦 Trained weights](https://github.com/MBalajiSakthivel007/G110_Github/releases)** ·
**[🗂️ Dataset](https://app.roboflow.com/mbalajis-workspace-g2gae/g110-beach-people-final/2)**

</div>

---

## 📌 The problem

Lifeguards watch crowded beaches by eye for hours, in bright glare. They need to know **how many people are in
the water** and where crowds are building up — but far swimmers are only 15–40 pixels tall on a camera, people
overlap, and umbrellas, boats and waves look like people. Ordinary person detectors cannot say whether someone is
on land or in the water.

**Our solution:** a two-class detector (*person on land* / *person in water*) trained on footage from fixed beach
CCTV, which produces live land / water head-counts for every frame. We built our own dataset and compared
**five deep detectors under one identical protocol** to find the best design for this task.

---

## 🏗️ Architecture

![Overall system architecture](Phase2/Architecture/01_Overall_System_Architecture.png)

The initial Phase I design and the architecture of each detector are in [`Phase2/Architecture/`](Phase2/Architecture).

---

## 🏆 Results

Same 399 test frames (4,546 people on land + 7,372 people in water) for every model.

| Model | Family | mAP@50 | mAP@50:95 | AP50 water | F1 | Speed |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **RT-DETR-L** | detection transformer, no NMS | **80.94** | **44.83** | **80.23** | 78.42 | **4.64 FPS** |
| 🥈 **Faster R-CNN R101-FPN** | two-stage | 79.24 | 41.87 | 74.59 | **79.83** | 1.30 FPS |
| 🥉 **Faster R-CNN R50-FPN** | two-stage, lighter backbone | 77.50 | 40.15 | 73.00 | 78.47 | 1.70 FPS |
| **YOLOv8x-P2** | one-stage + small-object head | 77.28 | 41.42 | 75.51 | 76.96 | 2.58 FPS |
| **RetinaNet R50-FPN** | one-stage, Focal Loss | 73.66 | 33.57 | 71.47 | 72.94 | 1.64 FPS |

**Highlights**

- **RT-DETR-L** is the most accurate, the best on swimmers and the fastest — the recommended model for a live system.
- **Faster R-CNN R101-FPN** has the best F1, the best accuracy on people on land and the lowest counting error.
- **Land and water are almost never confused** (below 1 % for every model).
- **Counting error of only 1–3 people per frame** on scenes with up to 50 people.
- **Fine-tuning adds 14–22 points** over a plain COCO detector; denoising hurts, CLAHE and test-time augmentation help some models.

---

## 🧠 The five detectors

| # | Detector | Framework | Owner | Documentation |
|---|---|---|---|---|
| 1 | Faster R-CNN R101-FPN | Detectron2 | M. Balaji Sakthivel | [📘 Guide](Phase2/Detectron2/Detectron2_FasterRCNN_R101_Documentation.md) · [📁 Files](Phase2/Detectron2) |
| 2 | Faster R-CNN R50-FPN | Detectron2 | Bhargava Ram Kaza | [📘 Guide](Phase2/FasterRCNN/FasterRCNN_R50_Documentation.md) · [📁 Files](Phase2/FasterRCNN) |
| 3 | YOLOv8x-P2 | Ultralytics | R. Shanmugham | [📘 Guide](Phase2/YOLO/YOLOv8x_P2_Documentation.md) · [📁 Files](Phase2/YOLO) |
| 4 | RT-DETR-L | Ultralytics | Chitturi Venkata Chaitanya | [📘 Guide](Phase2/RT-DETR/RTDETR_L_Documentation.md) · [📁 Files](Phase2/RT-DETR) |
| 5 | RetinaNet R50-FPN | Detectron2 | Shaik Ibbraheem | [📘 Guide](Phase2/RetinaNet/RetinaNet_R50_Documentation.md) · [📁 Files](Phase2/RetinaNet) |

Each model folder contains its documentation, Kaggle notebooks and an `Output/` folder with charts, sample frames,
all metrics and the output videos.

---

## 🗂️ Dataset — G110-BEACH-PEOPLE-FINAL

Built by the team from two public, fixed beach webcams — **White Bay, British Virgin Islands** (crowded) and
**Sunny Isles, USA** (far, small people).

| Split | Frames | People | On land | In water |
|---|---:|---:|---:|---:|
| Train | 3,454 | 132,027 | 57,634 | 74,393 |
| Validation | 782 | 31,443 | 13,558 | 17,885 |
| Test | 399 | 11,918 | 4,546 | 7,372 |

Split **by clip** (test clips from a different day), test set completed by eye, land / water decided from each
person's foot point and the camera's waterline. → [Open on Roboflow](https://app.roboflow.com/mbalajis-workspace-g2gae/g110-beach-people-final/2)

---

## 🚀 Quick start

1. Open a notebook from `Phase2/<model>/Notebooks/` on **Kaggle** with **GPU T4** and **Internet** enabled.
2. Add your Roboflow API key as a Kaggle secret named **`ROBOFLOW_API_KEY`** (Add-ons → Secrets).
3. **Run all** — the notebook downloads the dataset, trains, picks the best checkpoint and threshold on validation,
   evaluates on the test set and writes all charts, metrics and videos.
4. Want to skip training? Download the trained weights from **[Releases](https://github.com/MBalajiSakthivel007/G110_Github/releases)**.

---

## 📁 Repository structure

```
G110_Github/
├── README.md
├── docs/
│   └── Phase2_Project_Documentation.md   ← full project documentation
└── Phase2/
    ├── Architecture/    system and model architecture diagrams
    ├── Report/          project report (PDF)
    ├── References/      literature survey — 20 IEEE papers with links
    ├── Detectron2/      Faster R-CNN R101-FPN
    ├── FasterRCNN/      Faster R-CNN R50-FPN
    ├── YOLO/            YOLOv8x-P2
    ├── RT-DETR/         RT-DETR-L
    └── RetinaNet/       RetinaNet R50-FPN
```

---

## 👥 Team G110

| Register number | Name |
|---|---|
| CB.SC.U4CSE23213 | M. Balaji Sakthivel |
| CB.SC.U4CSE23214 | Chitturi Venkata Chaitanya |
| CB.SC.U4CSE23257 | R. Shanmugham |
| CB.SC.U4CSE23711 | Bhargava Ram Kaza |
| CB.SC.U4CSE23742 | Shaik Ibbraheem |

**Guide:** Ms. R. Sujee, Assistant Professor, Department of Computer Science and Engineering
**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore
**Course:** 23CSE498 — Project Phase II

---

## 📚 References

The literature survey — 20 IEEE papers, grouped by model, with official links — is in
[`Phase2/References/Literature_Survey_Papers.md`](Phase2/References/Literature_Survey_Papers.md).

## 📄 License

Released under the [MIT License](LICENSE). The beach webcam footage belongs to its owners and was used for research only.
