import cv2
import sys
import os
import argparse
import numpy as np

# Ensure modules can be imported from the root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from detection.yolo_detector import YOLOv8Detector
from density.density_estimator import DensityEstimator
from motion.flow_calculator import MotionAnalysis
from anomaly.risk_engine import RiskEngine, GridRiskEngine
from preprocessing.processor import VideoProcessor

def run_preprocessing(video_path):
    print("[INFO] Running Module 0: Video Preprocessing (CLAHE, Histogram Eq)")
    processor = VideoProcessor(target_size=(800, 600), apply_clahe=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"[ERROR] Cannot open {video_path}"); return

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Original vs Preprocessed side-by-side
        orig = cv2.resize(frame, (800, 600))
        processed = processor.process_frame(frame)
        
        cv2.putText(orig, "Original Feed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(processed, "Preprocessed feed (CLAHE / Resize)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        combined = np.hstack((orig, processed))
        combined = cv2.resize(combined, (1200, 450))
        
        cv2.imshow("Module Demo - Video Preprocessing", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

def run_detection(video_path):
    print("[INFO] Running Module 1: YOLOv8 Person Detection")
    detector = YOLOv8Detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"[ERROR] Cannot open {video_path}"); return

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (800, 600))
        bboxes, count, annotated = detector.detect(frame)
        
        cv2.putText(annotated, f"Detected People: {count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Module Demo - YOLO Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

def run_density(video_path):
    print("[INFO] Running Module 2: Gaussian Density Heatmap (CSRNet Hybrid)")
    detector = YOLOv8Detector()
    estimator = DensityEstimator(frame_size=(800, 600))
    from main import CSRNetEstimator
    csrnet = CSRNetEstimator()
    
    cap = cv2.VideoCapture(video_path)
    
    ema_count = None
    ema_density = None
    ema_alpha = 0.15
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (800, 600))
        bboxes, yolo_count, _ = detector.detect(frame)
        
        # Hybrid Density Logic for the demo
        if csrnet.ready:
            raw_map, csrnet_count = csrnet.predict(frame)
            if raw_map is not None:
                density_map = raw_map
                # Calibrated dense crowd scaler (Threshold out noise for sparse crowds)
                scaled_csr_volume = int(max(0, csrnet_count - 1.5) * 60)
                
                # If YOLO finds few people and CSRNet is low, rely on YOLO.
                # If huge dense crowd, rely on scaled CSRNet volume.
                if yolo_count < 10 and scaled_csr_volume < 40:
                    count = yolo_count
                else:
                    count = max(yolo_count, scaled_csr_volume)
                    
                density_pct = float(min(100.0, count * 0.25))
            else:
                density_pct, density_map, _, _ = estimator.estimate_density(bboxes, frame)
                count = yolo_count
        else:
            density_pct, density_map, _, _ = estimator.estimate_density(bboxes, frame)
            count = yolo_count

        
        # Normalize and colorize the heatmap
        d_max = density_map.max()
        if d_max < 0.0001:  # Cutoff for empty frame decay noise
            norm_map = density_map * 0
        else:
            norm_map = density_map / d_max
        heatmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame (alpha mask so 0-density is transparent)
        alpha = np.clip(norm_map * 2.5, 0.0, 1.0)
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        overlay = (frame.astype(np.float32) * (1.0 - alpha_3ch * 0.6) + 
                   heatmap_color.astype(np.float32) * (alpha_3ch * 0.6)).astype(np.uint8)
        
        # Smooth UI Metrics
        if ema_count is None:
            ema_count = float(count)
            ema_density = density_pct
        else:
            ema_count = (ema_alpha * count) + ((1 - ema_alpha) * ema_count)
            ema_density = (ema_alpha * density_pct) + ((1 - ema_alpha) * ema_density)
            
        cv2.putText(overlay, f"Density: {ema_density:.2f}% | Count: {int(ema_count)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Module Demo - Density Heatmap", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

def run_motion(video_path):
    print("[INFO] Running Module 3: Sparse Optical Flow (Motion)")
    motion = MotionAnalysis()
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (800, 600))
        score, overlay, _ = motion.estimate_motion(frame)
        
        cv2.putText(overlay, f"Motion Score: {score:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Module Demo - Motion Tracking", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

def run_combined(video_path):
    print("[INFO] Running Module 4: Integrated Anomaly Risk Engine")
    detector = YOLOv8Detector()
    estimator = DensityEstimator(frame_size=(800, 600))
    motion = MotionAnalysis()
    risk_engine = GridRiskEngine()
    
    from main import CSRNetEstimator
    csrnet = CSRNetEstimator()
    
    from visualization.dashboard import VisualizationDashboard
    dashboard = VisualizationDashboard()
    
    from alert.telemetry_engine import AlertEngine
    alert_engine = AlertEngine()

    cap = cv2.VideoCapture(video_path)
    
    ema_count = None
    ema_density = None
    ema_motion = None
    ema_alpha = 0.15
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (800, 600))
        
        # 1. Detection
        bboxes, yolo_count, _ = detector.detect(frame)
        
        # 2. Density (Hybrid)
        if csrnet.ready:
            raw_map, csrnet_count = csrnet.predict(frame)
            if raw_map is not None:
                density_map = raw_map
                scaled_csr_volume = int(max(0, csrnet_count - 1.5) * 60)
                if yolo_count < 10 and scaled_csr_volume < 40:
                    count = yolo_count
                else:
                    count = max(yolo_count, scaled_csr_volume)
                    
                density_pct = float(min(100.0, count * 0.25))
                
                # Hybrid Grid Metrics
                _, _, grid_counts, grid_densities = estimator.estimate_density(bboxes, frame)
                gr, gc = grid_counts.shape
                h, w = raw_map.shape
                cell_h, cell_w = max(1, h // gr), max(1, w // gc)
                for r in range(gr):
                    for c in range(gc):
                        cell_sum = float(raw_map[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w].sum())
                        cell_vol = int(max(0, cell_sum - 0.16) * 60)
                        grid_counts[r, c] = max(grid_counts[r, c], cell_vol)
                        grid_densities[r, c] = min(100.0, grid_counts[r, c] * 0.12 * (gr * gc))
            else:
                density_pct, density_map, grid_counts, grid_densities = estimator.estimate_density(bboxes, frame)
                count = yolo_count
        else:
            density_pct, density_map, grid_counts, grid_densities = estimator.estimate_density(bboxes, frame)
            count = yolo_count
        
        # 3. Motion
        motion_score, _, grid_motions = motion.estimate_motion(frame)
        
        # 4. Risk Fusion
        risk_level, color, anomaly_score, grid_results = risk_engine.assess_risk(
            count, density_pct, motion_score, grid_counts, grid_densities, grid_motions
        )
        
        # 5. Dashboard Render
        final_vis = dashboard.render(
            original_frame   = frame,
            bboxes           = bboxes,
            count            = count,
            density_pct      = density_pct,
            heatmap_overlay  = density_map,
            motion_score     = motion_score,
            risk_level       = risk_level,
            risk_color       = color,
            grid_results     = grid_results
        )
        
        # Smooth the UI metrics
        if ema_count is None:
            ema_count = float(count)
            ema_density = density_pct
            ema_motion = motion_score
        else:
            ema_count = (ema_alpha * count) + ((1 - ema_alpha) * ema_count)
            ema_density = (ema_alpha * density_pct) + ((1 - ema_alpha) * ema_density)
            ema_motion = (ema_alpha * motion_score) + ((1 - ema_alpha) * ema_motion)
        
        # Local UI Overlay (Side Panel Sim)
        cv2.rectangle(final_vis, (0, 0), (280, 220), (0, 0, 0), -1)
        cv2.putText(final_vis, f"COUNT: {int(ema_count)}", (10, 30), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(final_vis, f"DENSITY: {ema_density:.1f}%", (10, 70), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(final_vis, f"MOTION: {ema_motion:.2f}", (10, 110), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(final_vis, f"RISK: {risk_level}", (10, 170), 1, 2, color, 3)
        
        # 6. Module 5 (Telemetry Dispatch)
        alert_engine.process_risk(risk_level, ema_count, ema_density, ema_motion, anomaly_score, final_vis)
        
        cv2.imshow("Module Demo - Combined Dashboard", final_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Crowd Analysis - Module Demo for Screenshots")
    parser.add_argument("mode", choices=["preprocessing", "detection", "density", "motion", "dashboard"], 
                        help="Select which module to demo")
    parser.add_argument("--source", default="videos/test.mp4", help="Video file source")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"[ERROR] Video source not found: {args.source}")
        sys.exit(1)
        
    if args.mode == "preprocessing": run_preprocessing(args.source)
    elif args.mode == "detection": run_detection(args.source)
    elif args.mode == "density": run_density(args.source)
    elif args.mode == "motion": run_motion(args.source)
    elif args.mode == "dashboard": run_combined(args.source)
