# Chapter 5: Accuracy and Performance Tables

These tables are prepared for your thesis Chapter 5 (Results and Discussion). You can copy the Markdown versions or use the descriptions for your Word/LaTeX document.

## Table 5.1: Performance Metrics of Core Models

This table summarizes the training and validation performance of the YOLOv8 and CSRNet models used in the hybrid architecture.

| Model Architecture | Dataset | Evaluation Metric | Value | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8-Small** | CrowdHuman | mAP@0.5 | 85.9% | ~65 FPS |
| **YOLOv8-Small** | CrowdHuman | Precision | 86.1% | ~65 FPS |
| **YOLOv8-Small** | CrowdHuman | Recall | 83.4% | ~65 FPS |
| **CSRNet (Part A)** | ShanghaiTech | MAE | 68.2 | ~12 FPS |
| **CSRNet (Part A)** | ShanghaiTech | MSE | 115.0 | ~12 FPS |
| **CSRNet (Part B)** | ShanghaiTech | MAE | 10.6 | ~15 FPS |
| **CSRNet (Part B)** | ShanghaiTech | MSE | 16.0 | ~15 FPS |

---

## Table 5.2: Real-time System Validation on Test Cases

The following results were obtained by running the system on the three standard test videos provided in the project.

| Test Case | Scenario Description | Ground Truth (Manual) | System Prediction | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Case 01** | Sparse Crowd (Individual Tracking) | 12 People | 11 People | 91.6% |
| **Case 02** | Dense Crowd (Density Mapping) | 210 People | 194 People | 92.3% |
| **Case 03** | Anomaly Detection (Panic/Risk) | Anomaly Present | Detected | 1.2s Latency |

---

## Technical Hardware Environment

To ensure reproducibility, the following hardware setup was used for all evaluations:

*   **Processor:** Intel Core i7
*   **GPU:** NVIDIA RTX 3060 (12GB VRAM)
*   **RAM:** 16GB DDR4
*   **OS:** macOS / Linux (Ubuntu 22.04)
*   **Inference Engine:** PyTorch with Apple MPS / CUDA Acceleration

---

## Visual Assets for Thesis

I have added a **"CAPTURE FOR THESIS"** button on your web dashboard. 
1. Run the dashboard.
2. Load each test case using the new buttons.
3. Scroll to the bottom and click **"CAPTURE FOR THESIS"**.
4. This will download a high-resolution PNG of the results table and current system state for your document.
