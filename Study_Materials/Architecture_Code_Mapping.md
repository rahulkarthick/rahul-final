# System Architecture to Code Mapping — Viva Study Guide

This document maps the exact 4 layers of your Architecture Diagram directly to the Python code files in your project. Have this open during your presentation so when the examiner points to a box on your diagram and asks *"Where is this in the code?"*, you can immediately answer them.

---

## 🟢 LAYER 1: INGESTION LAYER
*This layer handles getting the video and making it ready for the AI.*

1. **"Raw Video Source" & "Video Reader Thread"**
   * **File:** `web/app.py`
   * **Where to point:** `def video_reader_thread()`
   * **What to say:** "The video ingestion happens asynchronously in Flask. This thread's only job is to read raw MP4 or Webcam frames using OpenCV `cap.read()` and push them into a fast thread-safe queue at the video's native FPS."

2. **"Preprocessing Stage"**
   * **File:** `preprocessing/processor.py`
   * **Where to point:** `class VideoProcessor`
   * **What to say:** "Before hitting the AI, we normalize the data. We use OpenCV to standardise the frame size to exactly 800x600 pixels to prevent memory crash and apply optional CLAHE processing to enhance contrast in dark lighting."

---

## 🟢 LAYER 2: CONCURRENCY CONTROL
*This layer proves you understand high-performance system engineering, not just AI.*

1. **"Async Inference Controller (Central Hub)"**
   * **File:** `main.py`
   * **Where to point:** `class AsyncInferenceThread(threading.Thread)` and the `run()` loop.
   * **What to say:** "This is the master orchestrator. Instead of blocking the UI, this thread runs completely in the background. It pops the preprocessed frame from the queue, runs all 3 AI models (YOLO, CSRNet, Optical Flow), and pushes the final fused data to the output queue."

2. **"MJPEG Streamer Engine"**
   * **File:** `web/app.py`
   * **Where to point:** `def _mjpeg_generator()` and `def video_feed()`
   * **What to say:** "Instead of freezing while the AI thinks, this Flask route streams the latest annotated frames via MJPEG (`multipart/x-mixed-replace`). It guarantees the web dashboard stays perfectly responsive at 25+ FPS."

---

## 🟢 LAYER 3: INTELLIGENCE AND FUSION
*This is the core Deep Learning layer mapping to your 3 models.*

1. **"YOLOv8 Detection (Sparse Object Logic)"**
   * **File:** `detection/yolo_detector.py`
   * **Where to point:** `self.model(frame, classes=[0])`
   * **What to say:** "This handles sparse crowds. We use YOLOv8 bounded strictly to Class 0 (Humans) running on MPS/CUDA hardware acceleration to extract precise localized bounding boxes."

2. **"CSRNet Density Map (Dense DCNN Logic)"**
   * **File:** `density/csrnet.py`
   * **Where to point:** `self.backend` (Dilated Convolutions)
   * **What to say:** "For dense overlaps where YOLO fails, we pass the frame through our CSRNet architecture. The dilated convolutions generate a spatial heatmap representing pure mathematical density without needing individual outlines."

3. **"Lucas-Kanade Flow (Kinetic Motion Vectors)"**
   * **File:** `motion/flow_calculator.py`
   * **Where to point:** `cv2.calcOpticalFlowPyrLK()`
   * **What to say:** "To measure panic, we calculate Sparse Optical Flow. It tracks the physical movement vectors of the crowd between frames to generate a real-time kinetic speed score."

**The Risk Fusion Engine (Anomaly Detection)**
* **File:** `anomaly/risk_engine.py`
* **Where to point:** `class GridRiskEngine` block -> Exponential Moving Average and formulas.
* **What to say:** "We don't trigger alarms manually. All 3 metrics (Count, Density, Motion) are fed into the Risk Engine. It applies **EMA Smoothing (Metric Noise Filtering)** to remove glitchy data. It compares the current motion to a dynamic baseline (`Mean + 2×Sigma`) to determine if a **Kinetic Panic** or **Congestion Crush** is truly occurring."

---

## 🟢 LAYER 4: OUTPUT AND FORENSIC
*This handles the final product UI and logging requirements.*

1. **"Security Dashboard UI"**
   * **File:** `visualization/dashboard.py`
   * **Where to point:** `def render_module_grid()`
   * **What to say:** "We use OpenCV alpha-blending to stitch the CSRNet heatmap, YOLO bounding boxes, and Optical flow lines together onto the visual UI to transmit to the frontend."

2. **"Forensic Snapshot Store & CSV Telemetry Database"**
   * **File:** `alert/telemetry_engine.py`
   * **Where to point:** `class AlertEngine`
   * **What to say:** "A true security system needs an audit trail. When the Risk Engine hits a CRITICAL threshold, this module automatically saves the exact metrics to a CSV and captures an image snapshot as forensic evidence of the anomaly."
