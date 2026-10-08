import cv2
import numpy as np

class MotionAnalysis:
    def __init__(self, smoothing_window=10, grid_size=(3, 3)):
        """
        Uses Memory-Optimized Sparse Optical Flow (Lucas-Kanade) for real-time inference.
        Farneback is too heavy for a multi-module system. This tracks corner features (like human heads).
        A rolling average is applied over `smoothing_window` frames to prevent noisy spikes.
        """
        self.prev_gray = None
        self.p0 = None
        # Rolling window buffer for motion smoothing
        self.smoothing_window = smoothing_window
        self._motion_history = []
        
        self.grid_rows, self.grid_cols = grid_size
        self._grid_motion_history = [[[] for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=15, blockSize=5)
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def estimate_motion(self, frame):
        """
        Calculates average magnitude of motion of tracked features, globally and grid-wise.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0
        grid_motions = np.zeros((self.grid_rows, self.grid_cols), dtype=float)
        overlay = frame.copy()
        
        # Re-initialize tracking points when features drop below threshold
        # This drastically improves kinetic accuracy by catching new pedestrians entering the scene
        if self.prev_gray is None or self.p0 is None or len(self.p0) < 75:
            self.prev_gray = gray_frame
            
            # ROI Masking (Region of Interest)
            # 1. Global Mask: Ignore the upper Sky and Buildings (Top 35%)
            h, w = gray_frame.shape
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_mask[int(h * 0.35):, :] = 255
            
            # 2. Targeted Billboard Mask: The giant digital video advertisement on the right 
            # side of the intersection extends much lower and confuses the AI. 
            # We surgically black out that specific grid sector without losing the street.
            roi_mask[:int(h * 0.65), int(w * 0.60):] = 0
            
            self.p0 = cv2.goodFeaturesToTrack(gray_frame, mask=roi_mask, **self.feature_params)
            
            # If no features found, just return 0
            if self.p0 is None:
                return 0.0, overlay, grid_motions
                
            return 0.0, overlay, grid_motions
            
        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_frame, self.p0, None, **self.lk_params)
        
        if p1 is not None and st is not None:
            # Select good points
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            
            # Calculate distance moved by each valid point
            magnitudes = []
            grid_mags_current = [[[] for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
            
            gh = frame.shape[0] / self.grid_rows
            gw = frame.shape[1] / self.grid_cols
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                dist = np.sqrt((a - c)**2 + (b - d)**2)
                
                # Filter stationary noise (crosswalk lines, road features)
                # Only draw and compute if the point moved at least slightly (0.5 px)
                if dist > 0.5:
                    magnitudes.append(dist)
                    # Draw motion trails only on moving objects
                    overlay = cv2.line(overlay, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 2)
                    overlay = cv2.circle(overlay, (int(a), int(b)), 3, (0, 0, 255), -1)
                    
                    c_idx = int(min(self.grid_cols - 1, max(0, a // gw)))
                    r_idx = int(min(self.grid_rows - 1, max(0, b // gh)))
                    grid_mags_current[r_idx][c_idx].append(dist)
                
            if magnitudes:
                # Raw kinetic magnitude reflects stable pedestrian motion
                raw_score = float(np.mean(magnitudes))
                # Rolling window smoothing — keeps displayed value stable
                self._motion_history.append(raw_score)
                if len(self._motion_history) > self.smoothing_window:
                    self._motion_history.pop(0)
                motion_score = float(np.mean(self._motion_history))
                
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    mags = grid_mags_current[r][c]
                    raw_val = float(np.mean(mags)) if mags else 0.0
                    self._grid_motion_history[r][c].append(raw_val)
                    if len(self._grid_motion_history[r][c]) > self.smoothing_window:
                        self._grid_motion_history[r][c].pop(0)
                    grid_motions[r, c] = float(np.mean(self._grid_motion_history[r][c]))
                
            # Update previous frame and points
            self.prev_gray = gray_frame.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            
        return motion_score, overlay, grid_motions

if __name__ == "__main__":
    print("[INFO] Testing Motion Analysis Module...")
    motion = MotionAnalysis()
    cap = cv2.VideoCapture("videos/test.mp4")
    if not cap.isOpened(): exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame = cv2.resize(frame, (640, 480))
        score, motion_overlay, _ = motion.estimate_motion(frame)
        
        cv2.imshow("Module 4 - Motion Analysis", motion_overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
