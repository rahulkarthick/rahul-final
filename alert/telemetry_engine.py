import os
import time
import csv
import cv2
from datetime import datetime

class AlertEngine:
    def __init__(self, log_dir="logs"):
        """
        Module 5: Telemetry & Alert Dispatcher.
        Automatically logs high-risk anomalies and saves snapshot evidence for Smart City security teams.
        """
        self.log_dir = log_dir
        self.snapshot_dir = os.path.join(log_dir, "snapshots")
        self.csv_path = os.path.join(log_dir, "anomaly_alerts.csv")
        self.cooldown_seconds = 8  # Wait 8 seconds before saving another alert to prevent disk spam
        self.last_alert_time = 0
        
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Initialize CSV structure if new
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Risk_Level", "Count", "Density_Pct", "Motion_Score", "Anomaly_Score_Pct", "Snapshot_File"])
                
    def process_risk(self, risk_level, count, density, motion, anomaly_score, frame_overlay):
        """
        Receives the output from Module 4 (Risk Fusion).
        If Moderate or High Risk is detected, writes telemetry directly to logs.
        Enforces a chronological cooldown.
        """
        if risk_level in ["MODERATE", "HIGH RISK"]:
            current_time = time.time()
            if current_time - self.last_alert_time >= self.cooldown_seconds:
                self.last_alert_time = current_time
                self._dispatch_alert(risk_level, count, density, motion, anomaly_score, frame_overlay)
                return True
        return False
        
    def _dispatch_alert(self, risk_level, count, density, motion, anomaly_score, frame):
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_filename = f"alert_{risk_level.replace(' ', '_')}_{timestamp_str}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
        
        # Save visual evidence 
        cv2.imwrite(snapshot_path, frame)
        
        # Log to database representation (CSV)
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp_str,
                risk_level,
                int(count),
                f"{density:.2f}",
                f"{motion:.2f}",
                f"{anomaly_score:.2f}",
                snapshot_filename
            ])
            
        print(f"\n[CRITICAL SYSTEM] Module 5 Dispatched Alert: {risk_level}! Evidential snapshot saved.")

if __name__ == "__main__":
    print("[INFO] Testing Module 5: Telemetry Alerts")
    import numpy as np
    engine = AlertEngine()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.process_risk("HIGH RISK", 450, 99.9, 15.5, 95.0, dummy_frame)
