import cv2
import numpy as np

class DensityEstimator:
    def __init__(self, frame_size=(800, 600), base_sigma_ratio=0.15):
        """
        frame_size       : (width, height) used for calculating density area
        base_sigma_ratio : σ = bbox_height * ratio (controls spread of Gaussian blobs)
        """
        self.frame_width, self.frame_height = frame_size
        self._prev_density = None
        self.temporal_alpha = 0.35 # EMA smoothing factor
        self.base_sigma_ratio = base_sigma_ratio

    def _gaussian_blob(self, density_map, cx, cy, sigma):
        """Adds a normalized Gaussian blob at (cx, cy) directly into density_map"""
        h, w = density_map.shape
        radius = int(3 * sigma) + 1
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x1 <= x0 or y1 <= y0: return

        xs = np.arange(x0, x1, dtype=np.float32) - cx
        ys = np.arange(y0, y1, dtype=np.float32) - cy
        xx, yy = np.meshgrid(xs, ys)
        blob = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        blob /= (blob.sum() + 1e-7)
        density_map[y0:y1, x0:x1] += blob

    def estimate_density(self, bboxes, frame, grid_size=(3, 3)):
        """
        Returns (density_pct, raw_density_map_float32, grid_counts, grid_densities)
        """
        h, w = self.frame_height, self.frame_width
        density_map = np.zeros((h, w), dtype=np.float32)

        grid_rows, grid_cols = grid_size
        grid_counts = np.zeros(grid_size, dtype=int)
        
        gh = h / grid_rows
        gw = w / grid_cols

        for (x1, y1, x2, y2) in bboxes:
            bh = max(1, y2 - y1)
            cx, cy = int((x1+x2)/2), int(y1 + bh*0.2) # Head center
            sigma = max(3.0, bh * self.base_sigma_ratio)
            self._gaussian_blob(density_map, cx, cy, sigma)
            
            # Grid assignment
            c_idx = int(min(grid_cols - 1, max(0, cx // gw)))
            r_idx = int(min(grid_rows - 1, max(0, cy // gh)))
            grid_counts[r_idx, c_idx] += 1

        # Temporal EMA smoothing
        if self._prev_density is not None:
            density_map = (self.temporal_alpha * density_map + 
                           (1 - self.temporal_alpha) * self._prev_density)
        self._prev_density = density_map.copy()

        raw_density_pct = float(min(100.0, len(bboxes) * 0.35))
        
        # Apply EMA to the percentage to prevent 0% drops
        if not hasattr(self, '_ema_density_pct'):
            self._ema_density_pct = raw_density_pct
        else:
            self._ema_density_pct = (self.temporal_alpha * raw_density_pct + 
                                     (1 - self.temporal_alpha) * self._ema_density_pct)
        
        density_pct = self._ema_density_pct
        
        # Grid density percentages
        # Calibrated scaling: ~31 people in a cell = 100% density (0.35 * 9)
        grid_densities = np.minimum(100.0, grid_counts * 3.15)
        
        # Apply slight final blur for smoothness
        density_map = cv2.GaussianBlur(density_map, (15, 15), 0)
        
        return density_pct, density_map, grid_counts, grid_densities



if __name__ == "__main__":
    print("[INFO] Testing Gaussian Density Estimator …")
    estimator = DensityEstimator(frame_size=(640, 480))
    cap = cv2.VideoCapture("videos/test.mp4")
    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (640, 480))

        # Use mock bboxes that look like real people (tall, varying heights)
        mock_bboxes = [
            (100, 80,  200, 320),
            (220, 100, 310, 380),
            (330, 150, 420, 400),
            (440, 60,  530, 350),
        ]
        density_pct, vis, _, _ = estimator.estimate_density(mock_bboxes, frame)
        cv2.putText(vis, f"Density: {density_pct:.1f}%  People: {len(mock_bboxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Gaussian Density Heatmap", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
