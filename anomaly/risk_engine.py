import numpy as np

class RiskEngine:
    def __init__(self, ema_alpha=0.15, history_size=150):
        """
        The hybrid fusion engine. Assesses Risk dynamically based on Count, Density %, and Motion Score.
        Includes an Exponential Moving Average (EMA) to eliminate noise and prevent false alarms.
        Uses a dynamic threshold based on the historical mean and standard deviation.
        """
        self.ema_alpha = ema_alpha
        self.history_size = history_size
        self.density_history = []
        self.motion_history = []

        self.ema_count   = 0.0
        self.ema_density = 0.0
        self.ema_motion  = 0.0
        self.initialized = False

    def assess_risk(self, count, density_pct, motion_score):
        """
        Calculates Anomaly Probability % using two factors:
        1. Congestion Risk: Extreme absolute density (Bottlenecks / Crushes).
        2. Kinetic Panic Risk: Sudden spikes in motion relative to historical baseline.
        """
        if not self.initialized:
            self.ema_count   = float(count)
            self.ema_density = density_pct
            self.ema_motion  = motion_score
            self.initialized = True
        else:
            self.ema_count   = (self.ema_alpha * count)       + ((1 - self.ema_alpha) * self.ema_count)
            self.ema_density = (self.ema_alpha * density_pct) + ((1 - self.ema_alpha) * self.ema_density)
            self.ema_motion  = (self.ema_alpha * motion_score)+ ((1 - self.ema_alpha) * self.ema_motion)
            
        self.density_history.append(self.ema_density)
        self.motion_history.append(self.ema_motion)
        if len(self.density_history) > self.history_size:
            self.density_history.pop(0)
            self.motion_history.pop(0)
            
        # 1. Congestion Risk (Static Thresholding)
        # Tuned for Review Demo: Safe up to 40%, extremely dangerous beyond 80%.
        if self.ema_density < 40.0:
            congestion_risk = 0.0
        else:
            # Scales linearly from 0% at 40 density to 100% at 80 density
            congestion_risk = min(100.0, ((self.ema_density - 40.0) / 40.0) * 100.0)
            
        # 2. Kinetic Panic Risk (Dynamic Thresholding)
        panic_risk = 0.0
        if len(self.motion_history) > 30:
            mean_motion = np.mean(self.motion_history)
            std_motion = np.std(self.motion_history)
            
            # The dynamic threshold (Variance gap shrank to exactly 1.0 + 2.0 * deviation)
            dynamic_motion_thresh = mean_motion + 1.0 + (2.0 * std_motion)
            
            if self.ema_motion > dynamic_motion_thresh:
                excess_motion = self.ema_motion - dynamic_motion_thresh
                panic_risk = min(100.0, (excess_motion / 2.0) * 100.0)
                
                # CRITICAL FIX FOR SPARSE CROWDS:
                # 1 person running in an empty street is NOT a "crowd stampede".
                # We scale down the panic risk if the frame is mostly empty (density < 20%).
                density_multiplier = min(1.0, self.ema_density / 20.0)
                panic_risk = panic_risk * density_multiplier
                
        # Total Anomaly Score is whichever is more severe: The Crush or The Panic
        anomaly_score = max(congestion_risk, panic_risk)
        
        # Determine Risk Label based on Anomaly Score
        if anomaly_score > 75.0:
            # Extreme anomaly
            return "HIGH RISK", (0, 0, 255), anomaly_score
        elif anomaly_score > 40.0:
            # Getting dangerous
            return "MODERATE", (0, 255, 255), anomaly_score
        else:
            # Steady walking or empty road
            return "SAFE", (0, 255, 0), anomaly_score

class GridRiskEngine:
    def __init__(self, ema_alpha=0.15, history_size=150, grid_size=(3, 3)):
        """
        Maintains risk engines for overall frame AND individual grid cells.
        grid_size: (rows, cols)
        """
        self.grid_rows, self.grid_cols = grid_size
        self.grid_engines = [[RiskEngine(ema_alpha, history_size) for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        self.overall_engine = RiskEngine(ema_alpha, history_size)

    def assess_risk(self, count, density_pct, motion_score, grid_counts=None, grid_densities=None, grid_motions=None):
        """
        Assesses both overall risk and per-cell risk.
        Returns: (overall_level, overall_color, overall_anomaly, grid_results)
        Where grid_results is a 2D list matching the grid layout containing (level, color, anomaly_score)
        """
        overall_level, overall_color, overall_anomaly = self.overall_engine.assess_risk(count, density_pct, motion_score)
        
        grid_results = []
        if grid_counts is not None and grid_densities is not None and grid_motions is not None:
            for r in range(self.grid_rows):
                row_risks = []
                for c in range(self.grid_cols):
                    # Guard against potentially missing or improperly shaped arrays (defensive programming)
                    gc = grid_counts[r][c] if hasattr(grid_counts[r], "__len__") else grid_counts[r * self.grid_cols + c]
                    gd = grid_densities[r][c] if hasattr(grid_densities[r], "__len__") else grid_densities[r * self.grid_cols + c]
                    gm = grid_motions[r][c] if hasattr(grid_motions[r], "__len__") else grid_motions[r * self.grid_cols + c]
                    
                    level, color, score = self.grid_engines[r][c].assess_risk(gc, gd, gm)
                    row_risks.append((level, color, score))
                grid_results.append(row_risks)
                
        return overall_level, overall_color, overall_anomaly, grid_results


if __name__ == "__main__":
    # Test Module
    print("[INFO] Testing Anomaly Detector Fusion")
    engine = RiskEngine()
    print("Test 1 (Low Cnt, Low Mot, Low Den):", engine.assess_risk(10, 20.0, 0.5)[0])
    print("Test 2 (High Cnt, Low Mot, High Den):", engine.assess_risk(80, 75.0, 0.5)[0])
    print("Test 3 (High Cnt, High Mot, High Den):", engine.assess_risk(100, 80.0, 3.5)[0])
