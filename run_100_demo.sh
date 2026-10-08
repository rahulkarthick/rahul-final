#!/bin/bash

# Hybrid Crowd Analysis - 100% Review Demo Script
# Usage: ./run_100_demo.sh

echo "=========================================================="
echo "   HYBRID CROWD ANALYSIS - 100% FINAL EXECUTION DEMO      "
echo "=========================================================="
echo ""
echo "PHASE 1: Local VS Code Step-by-Step Visualization"
echo "----------------------------------------------------------"
echo "This will securely open the individual OpenCV module windows"
echo "on your local system (YOLO, Density, and Optical Flow) to"
echo "demonstrate exactly how the AI processes frames."
echo ""
echo "NOTE: Make sure a valid video exists at videos/test.mp4,"
echo "or it will default to your webcam."
echo ""
echo "-> Press 'q' inside any of the video windows to close them"
echo "   and proceed to Phase 2."
echo ""

# Run the local CV2 step by step windows
python3 demo_step_by_step.py --source videos/test.mp4

echo ""
echo "=========================================================="
echo "PHASE 2: Full Web Dashboard & Anomaly Engine"
echo "----------------------------------------------------------"
echo "Starting Flask Server... Please click the link below when ready."
echo "http://127.0.0.1:5001"
echo "----------------------------------------------------------"

# Once closed, boot the actual Flask dashboard which has the reverted 16:9 large viewer
python3 web/app.py
