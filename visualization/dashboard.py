import cv2
import numpy as np
from datetime import datetime

class VisualizationDashboard:
    def __init__(self):
        """
        Takes outputs from all modules and combines them into a single professional dashboard frame.
        """
        pass

    def render(self, original_frame, bboxes, count, density_pct,
               heatmap_overlay, motion_score, risk_level, risk_color, csrnet_active=False, grid_results=None):
        """
        Renders YOLO bounding boxes and density heatmap onto the frame.
        heatmap_overlay is expected to be a raw float32 density map.
        """
        fh, fw = original_frame.shape[:2]
        dashboard_frame = original_frame.copy()

        # ─── 1. Process Density Heatmap ──────────────────────────────────
        if heatmap_overlay is not None:
            # Normalize map to 0-1 range for visualization
            # CSRNet might return negative values or very small floats
            heatmap_overlay = np.maximum(heatmap_overlay, 0) # Remove negative values
            d_max = heatmap_overlay.max()
            if d_max < 0.0001:
                norm_map = heatmap_overlay * 0
            else:
                norm_map = heatmap_overlay / d_max
            
            # Create color heatmap (JET)
            heat_u8 = (norm_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            
            if heatmap.shape[:2] != (fh, fw):
                heatmap = cv2.resize(heatmap, (fw, fh))
                norm_map = cv2.resize(norm_map, (fw, fh))

            # Alpha mask: heatmap is more opaque where density is higher
            # We want a subtle but clear visualization
            alpha = np.clip(norm_map * 2.5, 0.0, 1.0) # Boost for visibility
            alpha_3ch = np.stack([alpha] * 3, axis=-1)
            
            # Blend: Frame * (1-alpha) + Heatmap * (alpha * strength)
            strength = 0.55
            dashboard_frame = (original_frame.astype(np.float32) * (1.0 - alpha_3ch * strength) + 
                               heatmap.astype(np.float32) * (alpha_3ch * strength)).astype(np.uint8)

        # ─── 2. Draw Bounding Boxes ──────────────────────────────────────
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(dashboard_frame, (x1, y1), (x2, y2), risk_color, 2)

        # ─── 2.5 Draw Overlay Grids (Grid-Wise Anomaly) ─────────────────
        if grid_results is not None:
            grid_rows = len(grid_results)
            grid_cols = len(grid_results[0])
            gh = fh // grid_rows
            gw = fw // grid_cols
            
            overlay = dashboard_frame.copy()
            for r in range(grid_rows):
                for c in range(grid_cols):
                    g_level, g_color, _ = grid_results[r][c]
                    x1, y1 = c * gw, r * gh
                    x2, y2 = (c + 1) * gw, (r + 1) * gh
                    
                    # Fill semi-transparent color for non-safe zones to highlight anomaly heavily
                    if g_level in ["MODERATE", "HIGH RISK"]:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), g_color, -1)
                        
                    # Draw thick borders for clarity
                    border_thickness = 4 if g_level == "HIGH RISK" else 2
                    cv2.rectangle(dashboard_frame, (x1, y1), (x2, y2), g_color, border_thickness)
            
            # Blend overlay for the filled areas
            alpha = 0.25 # Semi-transparent
            cv2.addWeighted(overlay, alpha, dashboard_frame, 1 - alpha, 0, dashboard_frame)

        # ─── 3. High Risk Alert Flash ─────────────────────────────────────
        if risk_level == "HIGH RISK":
            if datetime.now().microsecond < 500000:
                cv2.rectangle(dashboard_frame, (0, 0), (fw-1, fh-1), (0, 0, 255), 12)

        return dashboard_frame

    def render_module_grid(self, original_frame, bboxes, heatmap_overlay, motion_overlay, final_dashboard):
        """
        Creates a 2x2 grid explicitly demonstrating the step-by-step intermediate modules.
        1. YOLO, 2. Heatmap, 3. Motion, 4. Fusion
        """
        # Cell 1: YOLO Detection
        cell1 = original_frame.copy()
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(cell1, (x1, y1), (x2, y2), (255, 50, 50), 2)
        cv2.rectangle(cell1, (0, 0), (450, 45), (0, 0, 0), -1)
        cv2.putText(cell1, "1. YOLOv8 Detection", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Cell 2: Density Heatmap
        fh, fw = original_frame.shape[:2]
        cell2 = original_frame.copy()
        if heatmap_overlay is not None:
            heatmap_overlay = np.maximum(heatmap_overlay, 0)
            d_max = heatmap_overlay.max()
            if d_max < 0.0001:
                norm_map = heatmap_overlay * 0
            else:
                norm_map = heatmap_overlay / d_max
            heat_u8 = (norm_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            if heatmap.shape[:2] != (fh, fw):
                heatmap = cv2.resize(heatmap, (fw, fh))
                norm_map = cv2.resize(norm_map, (fw, fh))
            alpha = np.clip(norm_map * 2.5, 0.0, 1.0)
            alpha_3ch = np.stack([alpha] * 3, axis=-1)
            cell2 = (original_frame.astype(np.float32) * (1.0 - alpha_3ch * 0.75) + 
                     heatmap.astype(np.float32) * (alpha_3ch * 0.75)).astype(np.uint8)
        cv2.rectangle(cell2, (0, 0), (450, 45), (0, 0, 0), -1)
        cv2.putText(cell2, "2. Density Heatmap", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Cell 3: Optical Flow
        cell3 = motion_overlay.copy() if motion_overlay is not None else original_frame.copy()
        cv2.rectangle(cell3, (0, 0), (450, 45), (0, 0, 0), -1)
        cv2.putText(cell3, "3. Optical Flow Motion", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Cell 4: Fusion Output
        cell4 = final_dashboard.copy()
        cv2.rectangle(cell4, (0, 0), (450, 45), (0, 0, 0), -1)
        cv2.putText(cell4, "4. AI Risk Fusion", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Resize for performance and UI layout (1280x960 total size)
        def resize_cell(img):
            return cv2.resize(img, (640, 480))
            
        top = np.hstack([resize_cell(cell1), resize_cell(cell2)])
        bottom = np.hstack([resize_cell(cell3), resize_cell(cell4)])
        return np.vstack([top, bottom])

