# Anomaly Detection (Risk Engine) - Complete Study & Modification Guide

This guide explains exactly how the `risk_engine.py` module works, the math behind it, and exactly which lines of code you can change to fine-tune the anomaly detection in the future.

## 1. How the Logic Works (The Concept)
We split Anomaly Detection into two distinct risks:
1. **Congestion Risk (Static):** Is the crowd packed too tightly? (Measured by Density %).
2. **Kinetic Panic Risk (Dynamic):** Is the crowd suddenly running/stampeding? (Measured by Optical Flow Motion, compared against previous frames).

The Final Anomaly % is whichever of these two risks is *higher*.

## 2. Step-by-Step Code Explanation

Inside `anomaly/risk_engine.py`, when `assess_risk(count, density_pct, motion_score)` is called:

**Step A: Exponential Moving Average (EMA)**
```python
self.ema_density = (self.ema_alpha * density_pct) + ((1 - self.ema_alpha) * self.ema_density)
```
*Why we do this:* Raw AI outputs flicker up and down constantly. EMA "smooths" the numbers so the dashboard percentages don't jump erratically.

**Step B: The Memory Buffer**
```python
self.density_history.append(self.ema_density)
```
*Why we do this:* We store the last 150 frames in a list. This allows the AI to "remember" what the normal walking speed has been for the last few seconds.

**Step C: Congestion Risk (Density Overload)**
```python
if self.ema_density < 60.0:
    congestion_risk = 0.0
else:
    congestion_risk = min(100.0, ((self.ema_density - 60.0) / 30.0) * 100.0)
```
*Why we do this:* If density is below 60%, it's safe (0% risk). If it goes from 60% to 90%, the risk scales linearly from 0% to 100%.

**Step D: Kinetic Panic Risk (Stampede Detection)**
```python
mean_motion = np.mean(self.motion_history)
std_motion = np.std(self.motion_history)
dynamic_motion_thresh = mean_motion + max(1.0, 2.0 * std_motion)
```
*Why we do this:* We find the Average Speed (`mean_motion`) and how much it varies (`std_motion`). 
The `dynamic_motion_thresh` is the panic trigger. If the crowd suddenly moves faster than this trigger, `panic_risk` shoots up.

---

## 3. How to Tweak & Change the Code in the Future

If the Anomaly Detection is too sensitive or not sensitive enough during your testing, open `anomaly/risk_engine.py` and modify these specific values:

### A. To change when "Congestion" triggers:
**Location:** Line 44 (`if self.ema_density < 60.0:`)
*   If the dashboard shows anomalies too early in dense crowds, change `60.0` to `75.0` (and adjust line 48 accordingly). It will only start warning you when the crowd is extremely packed.

### B. To make "Panic" triggers more/less sensitive:
**Location:** Line 60 (`dynamic_motion_thresh = mean_motion + max(1.0, 2.0 * std_motion)`)
*   **More Sensitive:** Change `2.0 * std_motion` to `1.0 * std_motion`. The crowd only needs to speed up *a little bit* to trigger an anomaly. 
*   **Less Sensitive:** Change `2.0 * std_motion` to `3.0 * std_motion` if you want it to trigger *only* for extreme running and sprinting.

### C. To make it "remember" longer:
**Location:** Line 4 (`def __init__(self, ema_alpha=0.15, history_size=150):`)
*   Change `history_size=150` to `300`. It will take longer to learn what "normal" is, making panic detection smoother over long periods but slightly slower to adapt to a new baseline walking speed.
