# PersonDetection_YOLOv9
Here is a **clean, professional, production-ready README** for your YOLO human-detection project, based fully on your current script  and the existing README draft you provided .

You can copy-paste this into your **README.md**.

---

# Human Detection with YOLO (Webcam / Video)

This project uses **Ultralytics YOLO (v8/v9)** to detect humans in real-time from a **webcam** or a **video file**.
It draws bounding boxes around people and displays a live **person count** on each frame.

---

## 🎯 Features

* Real-time human detection using YOLOv8/YOLOv9.
* Works with **webcam** or **local video**.
* Displays **bounding boxes**, **confidence scores**, and **live person count**.
* Simple, clean, and extensible Python script.
* Automatically filters detections to **person class only**.

---

## 📂 Project Structure

```
│
├── detect_humans.py      # Main human detection script
├── README.md             # Project documentation
├── yolov9t.pt / yolov8*.pt (optional)  # YOLO model weights
│
```

---

## 🛠 Requirements

* **Python 3.8+**
* Install dependencies:

```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

### ⚡ Optional (for GPU acceleration)

Install CUDA-enabled PyTorch before installing ultralytics:

👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 📸 Usage

### 1. Configure the script

In `detect_humans.py`:

```python
use_webcam = True         # True for webcam, False for video file
video_path = r"C:\path\to\video.mp4"
model = YOLO("yolov9t.pt")  # You may replace with yolov8n.pt / yolov8s.pt etc.
```



### 2. Run the script

```bash
python detect_humans.py
```

---

## 🧠 How it Works

* YOLO runs inference on each frame.
* It extracts bounding boxes (`res.boxes.xyxy`, `conf`, `cls`) and filters only **class 0 → person**.
* The script draws green boxes and displays a counter using OpenCV.


---

## ⌨ Controls

* Press **Q** in the window to quit the detection loop.

---

## 🐛 Troubleshooting

| Issue                                 | Solution                                                             |
| ------------------------------------- | -------------------------------------------------------------------- |
| `Error: Could not open video source.` | Wrong webcam index or incorrect video path.                          |
| No detections                         | Use a larger model (`yolov8s.pt` / `yolov8m.pt`) or better lighting. |
| Low FPS                               | Install GPU-enabled PyTorch (CUDA).                                  |
|                                       |                                                                      |

---

## 🚀 Extensions (Easy Upgrades)

* Save output as annotated video using `cv2.VideoWriter`.
* Save person counts to CSV.
* Send alerts when number of persons crosses a threshold.
* Deploy on Jetson Nano / Raspberry Pi for edge-AI scenarios.

---

## 📄 License

This repository currently has **no license**.
Add MIT/GPL/Apache license if making the project public.

---

If you want, I can also:

✅ **Generate a more advanced README with GIF demos, installation badges**
✅ **Add code to save video output**
✅ **Add object tracking (BYTETracker / DeepSORT)**
✅ **Rewrite your script into OOP format**

Just tell me!

