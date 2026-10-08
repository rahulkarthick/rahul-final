# Hybrid Crowd Framework: Architecture Explanation (Viva Study Guide)

*Use this guide to structure your explanation of the system architecture during your thesis defense.*

---

## High-Level Introduction
**"Our architecture is divided into 4 distinct layers, starting from video capture down to the final dashboard."**

---

## 1. The Ingestion Layer (Getting the Video Ready)
* **What it does:** This is where the system takes in the raw video (from an MP4 file, webcam, or IP camera).
* **Preprocessing:** Before the AI even touches the video, we process it. We resize the frames to ensure the system runs quickly and apply techniques like **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to normalize the lighting. This ensures the AI can see clearly even in low-light conditions or bad weather.

## 2. The Concurrency Control Layer (Keeping it Real-Time)
* **What it does:** This layer handles the "plumbing" to ensure the video doesn't lag or crash.
* **How it works:** Because running three different AI models simultaneously is very heavy on the CPU/GPU, we use an **Async Inference Controller** (Asynchronous threading). This means the video reader continuously pulls frames in the background without waiting for the AI to finish. It passes the results to the **MJPEG Streamer** to broadcast the video to the web dashboard smoothly, preventing bottlenecks.

## 3. The Intelligence & Fusion Layer (The "Brain" of the System)
* **What it does:** This is where the heavy lifting happens. The frame is passed into three separate AI modules at the same time:
  1. **YOLOv8 Detection:** Counts individual people in sparse areas using bounding boxes.
  2. **CSRNet Density Map (DCNN):** Generates a thermal-style heatmap for extremely dense areas where YOLO fails to count due to overlapping bodies.
  3. **Lucas-Kanade Flow:** Calculates the optical flow to track kinetic movement (to detect running or panic).
* **The Risk Fusion Engine:** The outputs of all three models don't just act independently; they are fed into our custom Risk Engine. We apply **EMA Smoothing (Exponential Moving Average)** to filter out noise/glitches. The system evaluates two things simultaneously:
  * **Congestion Analysis:** Is the crowd getting crushed?
  * **Kinetic Panic Detection:** Is the crowd suddenly running faster than the historical baseline (Mean + 2 Standard Deviations)?

## 4. The Output & Forensic Layer (What the User Sees)
* **What it does:** Finally, the fused data is pushed out to three different places.
  1. **Security Dashboard UI:** The live web interface showing the colored risk alerts and the video feed.
  2. **CSV Telemetry Database:** A continuous log of the crowd metrics (count, density, motion) for historical analysis.
  3. **Forensic Snapshot Store:** If a "High Risk" anomaly is detected, the system automatically saves an evidence photo of that exact moment for later review by authorities.

---

### 💡 Key Viva Pro-Tips

*   **If they ask "What makes your architecture special?":** 
    Point directly to **Layer 2 (Concurrency)** and the **Risk Fusion Engine in Layer 3**. 
    *   *Answer:* "Standard systems run frame-by-frame and freeze when doing heavy AI. Our asynchronous architecture keeps the stream alive, while our Fusion Engine prevents false alarms by cross-referencing three different AI models."
*   **If they ask "Why use both YOLO and CSRNet?":**
    *   *Answer:* "YOLO is great for sparse crowds but fails in dense occlusions. CSRNet handles dense crowds but lacks individual precision. Using both gives us a hybrid, adaptive solution."
