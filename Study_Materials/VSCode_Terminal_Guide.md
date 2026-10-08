# Guide: Running the Hybrid Crowd Framework in VS Code Terminal

Follow these steps exactly to start the project for your review or testing.

## Step 1: Open VS Code Terminal
1. Open the project folder in VS Code.
2. Go to the top menu: **Terminal** -> **New Terminal**.
3. A terminal window will open at the bottom of your screen.

## Step 2: Activate the Virtual Environment
To ensure all AI libraries (PyTorch, YOLOv8) work correctly, you must activate the environment:
```bash
source crowd_env/bin/activate
```
*(Your terminal prompt should now show `(crowd_env)` at the beginning).*

## Step 3: (Optional) Install Dependencies
If you are running this for the first time on a new machine, run:
```bash
pip install -r requirements.txt
```

## Step 4: Choose Your Running Mode

### Mode A: Step-by-Step Module Review (Recommended for Viva)
Shows each module (Detection -> Density -> Motion) individually.
```bash
./review_demo.sh
```
*Note: Press **'q'** on your keyboard to close a module window and move to the next one.*

### Mode B: Full Web Dashboard (60% Milestone)
Starts the professional web UI directly.
```bash
./run_60_demo.sh
```
1. Wait for the server to start.
2. Open your browser to: `http://127.0.0.1:5001`

---

## 🛠️ Important Terminal Commands
* **Stop the project**: Press `Ctrl + C` in the terminal.
* **Fix Permission Denied error**: If the scripts don't run, type:
  ```bash
  chmod +x *.sh
  ```
* **Clean cache**: If you see errors, type:
  ```bash
  rm -rf __pycache__
  ```
