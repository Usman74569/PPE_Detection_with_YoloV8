# 🦺 PPE Detection using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)

---

## 📌 Overview

This project demonstrates **Personal Protective Equipment (PPE) detection** using the **YOLOv8 object detection model**.

It can detect helmets, vests, gloves, boots, and other PPE items from:

* **Images**
* **Videos**
* **Webcam streams**

**Applications:**

* Workplace safety monitoring
* Construction sites
* Factories and industrial environments

The implementation is provided as a **Jupyter Notebook**:
`PPE_Detection_with_YoloV8/PPE_Detection_using_yolov8.ipynb`.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Usman74569/PPE_Detection_with_YoloV8.git
cd PPE_Detection_with_YoloV8
```

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python matplotlib numpy
```

### 3. Open the Notebook

```bash
jupyter notebook PPE_Detection_using_yolov8.ipynb
```

> Or run it in **Google Colab** for free GPU training.

---

## 🛠 Dataset Setup

1. Upload your dataset (e.g., `Construction Site Safety.v30-raw-images_latestversion.yolov8.zip`) to Colab:

```python
from google.colab import files
uploaded = files.upload()
```

2. Unzip the dataset:

```python
!unzip -q "Construction Site Safety.v30-raw-images_latestversion.yolov8.zip" -d /content/ppe_dataset
```

3. Inspect dataset files:

```python
!ls /content/ppe_dataset
```

4. View or edit the dataset configuration file:

```python
with open('/content/ppe_dataset/data.yaml', 'r') as f:
    print(f.read())
```

5. Example `data.yaml` configuration:

```yaml
path: /content/ppe_dataset
train: train/images
val: valid/images
test: test/images

nc: 25
names: ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

roboflow:
  workspace: roboflow-universe-projects
  project: construction-site-safety
  version: 30
  license: CC BY 4.0
  url: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30
```

---

## 🏋️ Training YOLOv8

### 1. Initial Training

```python
from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train on dataset
model.train(data='/content/ppe_dataset/data.yaml', epochs=35, imgsz=640)
```

### 2. Continue Training (Optional)

```python
# Load previously trained weights
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Continue training for additional epochs
model.train(data='/content/ppe_dataset/data.yaml', epochs=10, imgsz=640)
```

> Notes:
>
> * Training logs and best weights are saved automatically in `runs/train`.
> * Using a **GPU** speeds up training significantly.

---

## 🖼 Image Inference

```python
results = model("/content/ppe_dataset/valid/images/Mask2_mov-11_jpg.rf.918a13fa7ce3fd15ed1d138a75751bd4.jpg")
results[0].show()  # Display detected objects
```

---

## 🎥 Video Inference

```python
from ultralytics import YOLO
from IPython.display import Video

def detect_ppe_on_video(model_path, input_video_path, conf=0.25):
    model = YOLO(model_path)
    results = model.predict(source=input_video_path, conf=conf, save=True)
    return results[0].path

model_path = '/content/runs/detect/train/weights/best.pt'
input_video = '/content/Untitled video - Made with Clipchamp.mp4'

output_video_path = detect_ppe_on_video(model_path, input_video)
print("Output video saved at:", output_video_path)

# Display output video
Video(output_video_path)
```

> The output video will include bounding boxes and PPE labels for detected objects.

---

## 📁 Project Structure

```
PPE_Detection_with_YoloV8/
├── PPE_Detection_using_yolov8.ipynb    # Main notebook
├── dataset/                             # PPE dataset folder
├── requirements.txt                     # Python dependencies
├── README.md                            # Project overview
```

---

## 📦 Requirements

* Python 3.x
* ultralytics (YOLOv8)
* OpenCV
* NumPy
* Matplotlib

> Optional versions for reproducibility:

```
ultralytics >= 8.0
opencv-python >= 4.7
numpy >= 1.24
matplotlib >= 3.7
```

---

## 🧠 Model Info

* Model: YOLOv8n pretrained
* Task: Object Detection (PPE)
* Output: Bounding boxes with PPE labels and confidence scores

---

## 🙋‍♂️ Author

**Usman Syed**
Civil Engineer
Exploring AI for real-world applications 🌿

---

## 📌 Notes

* Dataset must follow **YOLOv8 format** for annotations.
* Training is faster on systems with **GPU support**.
* You can deploy the project for **real-time PPE detection** using **Streamlit**, **Flask**, or **FastAPI**.
* Optional: Add sample images/videos to showcase detection results in the repository.

---


