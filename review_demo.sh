#!/bin/bash

# ==========================================================
# HYBRID CROWD ANALYSIS - MASTER REVIEW DEMO
# This script runs each module step-by-step for your review.
# ==========================================================

# 1. Setup Environment
echo "[1/5] Activating Environment..."
source crowd_env/bin/activate

echo ""
echo "--------------------------------------------------------"
echo "STEP 1: PERSON DETECTION (YOLOv8)"
echo "Action: Close the window (press 'q') to proceed to next module."
echo "--------------------------------------------------------"
python3 demo_modules.py detection

echo ""
echo "--------------------------------------------------------"
echo "STEP 2: DENSITY ESTIMATION (Gaussian Heatmap)"
echo "Action: Close the window (press 'q') to proceed to next module."
echo "--------------------------------------------------------"
python3 demo_modules.py density

echo ""
echo "--------------------------------------------------------"
echo "STEP 3: MOTION ANALYSIS (Optical Flow)"
echo "Action: Close the window (press 'q') to proceed to next module."
echo "--------------------------------------------------------"
python3 demo_modules.py motion

echo ""
echo "--------------------------------------------------------"
echo "STEP 4: INTEGRATED LOCAL DASHBOARD"
echo "Action: Close the window (press 'q') to proceed to the FINAL WEB DEMO."
echo "--------------------------------------------------------"
python3 demo_modules.py dashboard

echo ""
echo "--------------------------------------------------------"
echo "STEP 5: FINAL WEB-BASED PROFESSIONAL DASHBOARD"
echo "Action: The browser will open. Press Ctrl+C in this terminal when finished."
echo "--------------------------------------------------------"
echo "Starting Flask Server at http://127.0.0.1:5001 ..."
python3 web/app.py
