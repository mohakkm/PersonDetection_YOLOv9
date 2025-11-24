# PersonDetection_YOLOv9
Detect Humans with YOLO

Simple script that runs an Ultralytics YOLO model on a video or webcam and draws bounding boxes only for people, plus a running person count.

Files
- `detect_humans.py`: Main script. Configurable `use_webcam` and `video_path`, and `model` filename.
- `yolov9t.pt`, `yolov8n.pt`: Example model files (may be different in your workspace).
- `output_*.mp4`, `*.npy`, `*.csv`: example data/outputs in the repo.

Requirements
- Python 3.8+
- Packages:
  - `ultralytics`
  - `opencv-python`
  - `numpy`

Install dependencies (PowerShell):

```powershell
pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

Note: For GPU acceleration, install a matching `torch` build with CUDA support before running `ultralytics`. See https://pytorch.org for instructions.

Usage
1. Configure the script `detect_humans.py`:
   - Set `use_webcam = True` to use the webcam (default in this copy).
   - Set `use_webcam = False` and update `video_path` to use a video file.
   - Change `model = YOLO("yolov9t.pt")` to point to a different model file if you prefer (e.g. `yolov8n.pt`, `yolov8s.pt`).

2. Run the script (PowerShell):

```powershell
python detect_humans.py
```

Controls
- Press `q` in the display window to quit the program.

Behavior
- The script runs inference on each frame and filters detections to only draw boxes whose class is `person` (COCO class 0). It displays a `Persons: N` counter at the top-left of the frame.

Troubleshooting
- "Error: Could not open video source.": check `video_path` or webcam permission/index.
- No detections: try a larger model (e.g. `yolov8s.pt`) or use a clearer input stream.
- Performance is CPU-bound by default; to use GPU, ensure `torch` with CUDA is installed and that `ultralytics` detects the GPU.

Extending
- To save annotated video, wrap frames with `cv2.VideoWriter` and write `annotated_frame` to disk.
- To log timestamps and counts, append per-frame data to a CSV or NumPy array.

License
- No license specified. Add one if you intend to share this repository publicly.
