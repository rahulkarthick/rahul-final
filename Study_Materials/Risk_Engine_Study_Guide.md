# 🧠 Risk Engine (Anomaly Detection) Study Guide

This is your master study guide for the **Anomaly Detection** module, located purely in `anomaly/risk_engine.py`. Use this guide to easily digest the architecture, logic, and math behind your Hybrid Crowd Framework's anomaly alerting system.

---

## 1. How It Fits Into the Architecture (Risk Fusion)
**The Problem:** YOLO counts people, CSRNet estimates density, and Optical Flow measures speed/movement. But raw numbers don't tell you if people are in danger.
**The Solution (Risk Fusion):** The `GridRiskEngine` takes the raw outputs from all three AI models, **"fuses"** them together, and produces a single unified "Risk Level" (SAFE, MODERATE, HIGH RISK). 

It acts as the central brain that makes the final decision on whether to trigger an alarm.

---

## 2. EMA Smoothing (Metric Noise Filtering)

Computer vision models occasionally hallucinate (e.g., they mistakenly count a shadow as moving fast). If we trusted every single frame, the alarm would falsely trigger constantly ("glitchy data").

**How it works in your code:**
The system uses an **Exponential Moving Average (EMA)** to smooth out these abrupt spikes. It acts like a shock absorber.

*   **Formula in Code:** `ema_value = (0.15 * New_Value) + (0.85 * Old_EMA_Value)`
*   **What this means:** We only trust 15% of what we see in the *current split-second frame*. We rely on 85% of what we've seen *historically over the last few seconds*. If a short visual glitch happens, the 85% historical weight "pulls the number down" so the alarm doesn't go off. If the danger is real and sustained, the 15% starts stacking up until the alarm fires.

---

## 3. Congestion Analysis (The "Crush" Detector)

This part of the engine detects slow-moving, tightly packed crowds (bottlenecks).

*   **Logic:** It uses **Static Thresholding**. 
*   **How it works in your code:** It looks solely at the smoothed EMA density (`self.ema_density`).
    *   If density is below **35%**: Risk is `0.0` (Safe).
    *   If density goes above **35%**: Risk scales linearly up to 100% danger at 70% density.
*   **Real-world scenario:** A stadium exit gate where people are standing shoulder-to-shoulder, unable to walk forward.

---

## 4. Kinetic Panic Detection (The "Stampede" Detector)

This part detects sudden, erratic fleeing or running.

*   **Logic:** It uses **Dynamic Thresholding** (`Mean + 2 × Sigma`). You can't use a fixed speed limit because a marathon has naturally higher motion than a museum.
*   **How it works in your code:**
    1.  **Find Normal (`Mean`):** The system continuously calculates the average speed of the crowd over the last 30 frames.
    2.  **Find Normal Fluctuation (`Std / Sigma`):** It calculates the standard deviation (how much people speed up or slow down normally).
    3.  **The Threshold:** `dynamic_threshold = Mean + 1.0 + (2.0 * Sigma)`. 
    4.  **The Trigger:** If the current `ema_motion` breaks above that dynamically calculated threshold, it means a sudden anomaly (like a bomb scare or stampede) is happening right now.
*   **The Sparse Crowd Fix:** To prevent false alarms (like one solo jogger running across an empty street), the code scales down the "Panic Risk" if the density is very low (`density < 20%`).

---

## 5. The Final Fusion Decision

At the very end of `assess_risk()`, the engine has two scores: `congestion_risk` and `panic_risk`. 

How does it decide?
```python
# Total Anomaly Score is whichever is more severe: The Crush or The Panic
anomaly_score = max(congestion_risk, panic_risk)
```
It takes the **maximum** of the two. If `anomaly_score` is > 75, it throws a "HIGH RISK" (Red color) alarm. If it's > 40, it throws a "MODERATE" risk. Otherwise, it stays "SAFE".
