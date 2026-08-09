# 🦜 Cockatiel vs Budgerigar Detector

An object detection project built with **YOLOv8** and deployed with **Streamlit**. The model detects and classifies birds in images or videos as either a **Cockatiel** or a **Budgerigar**.

Built as part of Tuwaiq academy's Data Science and Ai Bootcamp.

## Project Overview

- **Dataset**: Combined and merged from two Roboflow Universe datasets (Cockatiel, Budgerigar)
- **Annotation**: Bounding boxes via Roboflow
- **Model**: YOLOv8n, trained on Google Colab (GPU)
- **Evaluation**: mAP50, precision, recall
- **Deployment**: Streamlit web app supporting both image and video input

## Results

| Metric | Score |
|---|---|
| mAP50 | 0.78 |
| Precision | 0.78 |
| Recall | 0.71 |

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| Cockatiel | 0.80 | 0.78 | 0.82 |
| Budgerigar | 0.76 | 0.64 | 0.74 |

## Repository Structure

```
.
├── app.py                  # Streamlit deployment app
├── requirements.txt        # Python dependencies
├── best.pt                 # Trained YOLOv8 model weights (not tracked in git, add manually)
├── YOLO_model_training.ipynb  # Training notebook (Colab)
└── README.md
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/LEOP305/YOLO_birds_classifier
   cd YOLO_birds_classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r Requirements.txt
   ```

3. the trained `best.pt` model is already in the project root (see [Training](#training) if you need to train your own).

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Training

Training was done in `YOLO_model_training.ipynb` on Google Colab using a GPU runtime:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=512,
    batch=16
)
```

## App Features

- **Image mode**: upload a photo, view annotated bounding boxes and per-detection confidence scores
- **Video mode**: upload a video, run frame-by-frame detection, preview and download the annotated result
- Adjustable confidence threshold via sidebar slider

