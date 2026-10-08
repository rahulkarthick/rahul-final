import matplotlib.pyplot as plt
import numpy as np
import os

# Create graphs directory if it doesn't exist
OUTPUT_DIR = "Thesis_Documentation/graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18
    })

def generate_fps_complexity_graph():
    """Graph 5.1: FPS vs Crowd Complexity"""
    # Simulate data
    complexity = np.linspace(0, 250, 100)
    fps = []
    
    for c in complexity:
        if c < 50:
            # Sparse state (YOLOv8s)
            fps.append(65 + np.random.normal(0, 2))
        elif c < 60:
            # Transition state
            fps.append(np.linspace(65, 15, 10)[int((c-50))]) # Simplified transition
        else:
            # Dense state (CSRNet)
            fps.append(15 + np.random.normal(0, 0.5))
            
    plt.figure(figsize=(10, 6))
    plt.plot(complexity, fps, color='#3b82f6', linewidth=3, label='System Throughput (FPS)')
    
    # Add threshold line
    plt.axvline(x=55, color='#ef4444', linestyle='--', alpha=0.7, label='Switching Threshold')
    
    plt.fill_between(complexity, fps, color='#3b82f6', alpha=0.1)
    
    plt.title("Graph 5.1: System Throughput vs. Crowd Complexity", pad=20)
    plt.xlabel("Crowd Density (Number of Personnel)")
    plt.ylabel("Inference Speed (Frames Per Second)")
    
    plt.annotate('Sparse State (YOLOv8s)', xy=(20, 68), xytext=(30, 75),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.annotate('Dense State (CSRNet)', xy=(150, 16), xytext=(160, 30),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.ylim(0, 90)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fps_vs_complexity.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/fps_vs_complexity.pdf")
    print(f"Saved: {OUTPUT_DIR}/fps_vs_complexity.png")

def generate_risk_score_graph():
    """Graph 5.2: EMA Risk Score over Time"""
    # Simulate data
    time = np.linspace(0, 120, 500)
    risk_score = np.zeros_like(time)
    
    for i, t in enumerate(time):
        if t < 65:
            # Normal baseline
            risk_score[i] = 0.15 + np.random.normal(0, 0.02)
        elif t < 70:
            # Rapid Surge (Anomaly)
            risk_score[i] = 0.15 + (t - 65) * 0.15 + np.random.normal(0, 0.03)
        elif t < 85:
            # High Risk Plateau
            risk_score[i] = 0.9 + np.random.normal(0, 0.04)
        else:
            # Recovery
            risk_score[i] = 0.9 - (t - 85) * 0.02 + np.random.normal(0, 0.02)
            
    risk_score = np.clip(risk_score, 0, 1.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, risk_score, color='#a855f7', linewidth=2.5, label='EMA Risk Index')
    
    # Thresholds
    plt.axhline(y=0.4, color='#f97316', linestyle=':', label='Moderate Risk Threshold')
    plt.axhline(y=0.75, color='#ef4444', linestyle=':', label='High Risk Threshold')
    
    plt.fill_between(time, risk_score, color='#a855f7', alpha=0.15)
    
    plt.title("Graph 5.2: EMA Risk Score Timeline (Anomaly Detection)", pad=20)
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Risk Index (0.0 - 1.0)")
    
    # Annotation
    plt.annotate('Panic/Anomaly Event Trigger', xy=(66, 0.3), xytext=(20, 0.5),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black'))

    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/risk_score_timeline.png", dpi=300)
    plt.savefig(f"{OUTPUT_DIR}/risk_score_timeline.pdf")
    print(f"Saved: {OUTPUT_DIR}/risk_score_timeline.png")

if __name__ == "__main__":
    setup_style()
    generate_fps_complexity_graph()
    generate_risk_score_graph()
