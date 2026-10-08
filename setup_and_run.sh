#!/bin/bash

echo "------------------------------------------------"
echo "🌟 HYBRID CROWD FRAMEWORK - FULL REPAIR & RUN"
echo "------------------------------------------------"

# 1. Kill any existing server on Port 5001
echo "[1/5] Cleaning up background processes..."
kill $(lsof -t -i:5001) > /dev/null 2>&1

# 2. Create/Refresh Virtual Environment
if [ ! -d "crowd_env" ]; then
    echo "[2/5] Creating Virtual Environment (crowd_env)..."
    python3 -m venv crowd_env
else
    echo "[2/5] Virtual Environment found."
fi

# 3. Activate and Install
echo "[3/5] Activating and Installing libraries..."
source crowd_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verification
echo "[4/5] Verifying Flask installation..."
if python3 -c "import flask" > /dev/null 2>&1; then
    echo "✅ Flask is ready!"
else
    echo "❌ ERROR: Flask failed to install. Please check your internet."
    exit 1
fi

# 5. Launch
echo "[5/5] Launching Professional Dashboard..."
echo "------------------------------------------------"
echo "🚀 ONCE RUNNING, CLICK THIS: http://127.0.0.1:5001"
echo "------------------------------------------------"
python3 web/app.py
