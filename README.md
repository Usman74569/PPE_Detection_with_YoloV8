
# 🦺 PPE Detection using YOLOv8

## 📌 Overview
This project demonstrates **Personal Protective Equipment (PPE) detection** using the YOLOv8 object detection model inside a Jupyter Notebook (`PPE_Detection_using_yolov8ipynb.ipynb`).  
It can detect safety helmets, vests, gloves, boots, and other PPE items from **images**, **videos**, or **webcam streams**.  
Useful for workplace safety monitoring in **construction sites**, **factories**, and **industrial environments**.

---

## 📦 Installation

1. **Clone this repository**

```bash
git clone https://github.com/Usman74569/PPE_Detection_using_yolov8.git
cd PPE_Detection_using_yolov8
````

2. **Install dependencies**

```bash
pip install ultralytics opencv-python matplotlib numpy
```

3. **Open Jupyter Notebook**

```bash
jupyter notebook PPE_Detection_using_yolov8ipynb.ipynb
```

---

## 🛠 Training

You can train the YOLOv8 model on your PPE dataset using the following Python code snippet inside your Jupyter Notebook or Python script:

```python
!pip install ultralytics --quiet
from ultralytics import YOLO

# Load the pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Start training on your dataset
model.train(data='/content/ppe_dataset/data.yaml', epochs=35, imgsz=640)
```

* Replace `/content/ppe_dataset/data.yaml` with the path to your dataset configuration file.
* Adjust `epochs` (number of training cycles) and `imgsz` (image input size) as needed.
* After training, the best weights and logs will be saved automatically in the `runs/train` directory.

---

```
