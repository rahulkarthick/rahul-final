# Hybrid Crowd Analysis Framework — Third Review / Viva Study Guide

This document is your **step-by-step presentation script**. It tells you exactly **what file to open**, **what code to point to**, and **what to say** during your third review (Viva). Follow this flow from Start to Finish to explain your whole project module by module.

---

## 🟢 Step 1: The User Interface (Web Dashboard)
**File to open:** `web/app.py`
**What it does:** Starts the web server and manages video streams.

*   **Code to highlight:** The `video_reader_thread()` and `result_collector_thread()` functions.
*   **What to explain:** 
    > "Our system uses a professional **3-thread architecture** via Flask. One thread dedicatedly reads frames at native FPS, a second thread runs the AI Inference Engine without blocking the UI, and a third thread streams the MJPEG result back to the web dashboard. This ensures the live feed never freezes even under heavy ML load."

---

## 🟢 Step 2: The Core Orchestrator
**File to open:** `main.py`
**What it does:** The central brain that connects all AI modules together.

*   **Code to highlight:** The `AsyncInferenceThread.run()` loop.
*   **What to explain:**
    > "Here is the master pipeline. Every frame from the dashboard passes through this sequence: 
    > 1. Preprocessing
    > 2. YOLOv8 Person Detection
    > 3. Density Estimation (CSRNet & Spatial Fallback)
    > 4. Visualization Overlay"

---

## 🟢 Step 3: Module 1 - Object Detection (YOLOv8)
**File to open:** `detection/yolo_detector.py` *(or explain from `main.py`)*
**What it does:** Identifies individual people with high accuracy.

*   **Code to highlight:** In `main.py`, show `bboxes, yolo_count, yolo_annotated = self.detector.detect(proc_frame)`.
*   **What to explain:**
    > "We utilize the YOLOv8-nano model for real-time person detection. It draws bounding boxes around easily visible individuals and gives us a baseline `yolo_count`."

---

## 🟢 Step 4: Module 2 - Density Estimation (CSRNet)
**File to open:** `density/csrnet.py`
**What it does:** The Deep Learning model that handles highly congested scenes where YOLO fails.

*   **Code to highlight:** Point to the `CSRNet` class, specifically the `self.frontend` and `self.backend` in the `__init__` function.
*   **What to explain:**
    > "This is our Congested Scene Recognition Network (CSRNet). 
    > It is built with a **VGG-16 frontend** to extract deep features, and a sequence of **Dilated Convolutions in the backend**. Dilated convolutions expand the model's receptive field so it can understand the scale of the crowd without losing resolution. The output is a Density Heatmap."

*   **Important Note:** Show `main.py` where we calculate `raw_density_volume = int(csrnet_count * 150)`.
    > "Because the CSRNet map can have very small fractional values, we scale the raw density map volume to compute a realistic crowd count metric, taking the maximum between the YOLO count and the normalized CSRNet mass."

---

## 🟢 Step 5: The Visualization / Rendering Engine
**File to open:** `visualization/dashboard.py`
**What it does:** Combines the video frame, bounding boxes, text, and the Heatmap into one visual result.

*   **Code to highlight:** The `render()` function and specifically the `Process Density Heatmap` section.
*   **What to explain:**
    > "Finally, we render the AI outputs back onto the video. We take the raw 2D density map from CSRNet, normalize its values (handling any negative numbers), and apply a OpenCV `COLORMAP_JET` to turn it into a red/blue heatmap. We then perform an alpha blend to overlay it smoothly on top of the original video."

---

## 🟢 Step 6: Module 3 - Anomaly Detection (Risk Engine)
**File to open:** `anomaly/risk_engine.py`
**What it does:** Calculates Anomaly % by separating normal crowd flow from dangerous panic or extreme congestion.

*   **Code to highlight:** The `assess_risk()` function, specifically `dyn_motion_thresh = mean + 1.5 * std`.
*   **What to explain:**
    > "Our anomaly detection uses a **Behavior-Aware Dynamic Threshold**. It doesn't just look at absolute numbers. 
    > If 100 people are crossing a road at a steady pace, the system learns this as the 'normal baseline walking speed' using a rolling statistical Mean and Standard Deviation. Because the speed is consistent, the **Kinetic Panic Risk** stays at 0%. 
    > It will ONLY trigger an anomaly if there is a **sudden spike** in motion out of nowhere (like a stampede or panic), OR if absolute density crosses the extreme safety limit (Congestion Crush). This prevents false alarms during normal heavy foot traffic."

---

## 💬 Quick Q&A Cheat Sheet (If the panel asks)

**Q: Where is the trained AI model stored?**
**A:** "The mathematical weights are stored in `models/csrnet/csrnet_model.pth`. It is a binary PyTorch checkpoint. The architecture code that uses these weights is in `density/csrnet.py`."

**Q: Why use both YOLO and CSRNet?**
**A:** "YOLO is excellent at detecting individuals in sparse areas, but it fails when people overlap entirely in heavy crowds. CSRNet solves this by predicting crowd density mathematically rather than looking for individual faces/shoulders."

**Q: Why do you scale the CSRNet count?**
**A:** "The model outputs a raw probability/density field. By integrating this field and applying a scaling multiplier, we map the activation energy to an approximate human count representation suitable for the dashboard UI."
