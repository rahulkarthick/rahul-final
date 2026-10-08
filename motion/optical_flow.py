import cv2
import numpy as np

class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None

    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(frame), 0.0

        # Calculate optical flow using Farneback
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Compute magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Visualization
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.prev_gray = gray
        avg_motion = np.mean(mag)

        return flow_rgb, avg_motion
