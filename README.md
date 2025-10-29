# **PPE Detection Using YOLOv8**

This project demonstrates **Personal Protective Equipment (PPE) detection** using the **YOLOv8 object detection model**. It detects items like helmets, vests, gloves, masks, and other PPE from images, videos, or live webcam streams.

Applications include:

* Workplace safety monitoring
* Construction sites
* Factories and industrial environments

The implementation is in a **Google Colab/Jupyter Notebook**: `PPE_Detection_using_yolov8.ipynb`.

---

## **Notebook Overview**

The notebook is divided into **logical sections**, each performing a specific task:

### 1. **Dataset Setup**

* Organizes dataset files into `train`, `valid`, and `test` folders.
* Defines dataset configuration in `data.yaml`:

  * Path to dataset
  * Number of classes (`nc`)
  * Class names (`names`)
  * Roboflow project information

This setup ensures YOLOv8 can correctly locate images and annotations for training.

---

### 2. **Installing Dependencies**

* Installs YOLOv8 via `ultralytics`
* Imports required Python packages (`ultralytics`, `IPython.display`, etc.)

**Purpose:** Prepare the environment for training and inference.

---

### 3. **Model Initialization**

* Loads a **pretrained YOLOv8n model** (`yolov8n.pt`).

**Purpose:**

* Provides a lightweight starting point with pretrained weights for faster convergence.
* Ensures that the model can leverage general object detection features before fine-tuning on PPE data.

---

### 4. **Training the Model**

* **Initial Training:**

  * Trains YOLOv8 on your PPE dataset using the configuration from `data.yaml`.
  * Typical parameters: `epochs=35`, `imgsz=640`.
* **Continued Training:**

  * Loads previously trained weights (`best.pt`)
  * Trains for additional epochs (`epochs=10`) to improve performance

**Purpose:**

* Adapt the YOLOv8 model to the specific PPE detection task.
* Save best-performing model weights automatically to `runs/train/weights/best.pt`.

---

### 5. **Image Inference**

* Performs PPE detection on a single image:

```python
results = model("valid/images/Mask2_mov-11_jpg.rf.918a13fa7ce3fd15ed1d138a75751bd4.jpg")
results[0].show()  # Display bounding boxes and labels
```

**Purpose:**

* Visualize the model’s predictions on validation/test images.
* Confirm that PPE items are correctly detected and labeled.

---

### 6. **Video Inference**

* Detect PPE on videos and save the output:

```python
from IPython.display import Video
output_video_path = detect_ppe_on_video(model_path, input_video)
Video(output_video_path)
```

**Purpose:**

* Apply the trained model to real-world scenarios, e.g., construction site video feeds.
* The output video contains bounding boxes, class labels, and confidence scores for all detected PPE items.

---

### 7. **Function Definitions**

* `detect_ppe_on_video(model_path, input_video_path, conf=0.25)`

  * Loads the model from `model_path`
  * Runs predictions on the input video
  * Saves the output video with detected objects

**Purpose:**

* Modular function for reusable video inference.
* Easily integrates into pipelines or other applications.

---

### 8. **Requirements**

* **Python version:** 3.x
* **Libraries:**

  * `ultralytics` (YOLOv8)
  * `opencv-python`
  * `numpy`
  * `matplotlib`
* Optional versions for reproducibility:

  ```
  ultralytics >= 8.0
  opencv-python >= 4.7
  numpy >= 1.24
  matplotlib >= 3.7
  ```

---

### 9. **Key Takeaways**

* **Model:** YOLOv8n (pretrained, fine-tuned on PPE dataset)
* **Task:** Object detection
* **Output:** Bounding boxes with PPE labels and confidence scores
* **Applications:** Real-time monitoring and automated workplace safety checks

---

### 10. **Author**

**Usman Syed** – Civil Engineer exploring AI for real-world applications 🌿

---

