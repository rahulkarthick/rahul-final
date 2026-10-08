import cv2
import numpy as np

class CrowdVisualizer:
    def __init__(self):
        # Professional color palette
        self.header_bg = (40, 40, 40)
        self.text_color = (255, 255, 255)
        self.accent_color = (0, 255, 0) # Green for count
        self.warning_color = (0, 165, 255) # Orange

    def create_dashboard(self, frame, yolo_frame, density_map, flow_frame, person_count, density_count):
        h, w = frame.shape[:2]
        
        # 1. Normalize and colorize density map
        density_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
        density_color = cv2.resize(density_color, (w, h))

        # 2. Add labels to each pane
        cv2.putText(yolo_frame, "YOLO: Object Detection", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(density_color, "CSRNet: Density Map", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(flow_frame, "Motion: Optical Flow", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # 3. Create Info Panel (Sidebar or Bottom)
        info_panel = np.zeros((h, w, 3), dtype=np.uint8)
        info_panel[:] = self.header_bg

        # Header
        cv2.rectangle(info_panel, (0, 0), (w, 60), (60, 60, 60), -1)
        cv2.putText(info_panel, "HYBRID CROWD ANALYSIS", (w//10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.9, self.accent_color, 2)

        # Statistics
        content_y = 120
        cv2.putText(info_panel, "CROWD STATISTICS", (40, content_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        
        content_y += 60
        cv2.putText(info_panel, f"YOLO COUNT: {person_count}", (40, content_y), cv2.FONT_HERSHEY_DUPLEX, 0.9, self.text_color, 2)
        
        content_y += 50
        cv2.putText(info_panel, f"DENSITY COUNT: {int(density_count)}", (40, content_y), cv2.FONT_HERSHEY_DUPLEX, 0.9, self.text_color, 2)

        # Confidence bar (simulated or based on delta)
        content_y += 40
        diff = abs(person_count - density_count)
        status = "CONSISTENT" if diff < 10 else "DISCREPANCY"
        status_color = self.accent_color if status == "CONSISTENT" else self.warning_color
        cv2.putText(info_panel, f"STATUS: {status}", (40, content_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Credits
        cv2.putText(info_panel, "MCA Final Year Project", (40, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # 4. Assemble Dashboard (2x2)
        top_row = np.hstack((yolo_frame, density_color))
        bottom_row = np.hstack((flow_frame, info_panel))
        dashboard = np.vstack((top_row, bottom_row))

        # Downscale for screen resolution
        target_width = 1280
        aspect_ratio = dashboard.shape[0] / dashboard.shape[1]
        target_height = int(target_width * aspect_ratio)
        dashboard_resized = cv2.resize(dashboard, (target_width, target_height))

        return dashboard_resized
