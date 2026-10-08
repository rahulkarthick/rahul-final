
import sys
import os
import cv2
import time
import threading
import queue
import numpy as np

from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename

# Ensure the root project is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from main import AsyncInferenceThread

# ─── Flask App ────────────────────────────────────────────────
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(ROOT, "videos")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ─── Shared State (protected by locks) ───────────────────────
_video_source       = None                # Starts idle until user loads a file
_video_source_lock  = threading.Lock()
_source_changed     = threading.Event()   # signals the reader thread

_latest_jpeg        = None                # latest processed JPEG bytes
_jpeg_lock          = threading.Lock()
_jpeg_ready         = threading.Event()   # signals new JPEG is available

_latest_metrics     = {
    "count": 0, "density": 0.0, "motion": 0.0,
    "risk": "SAFE", "color": "#00ff00", "csrnet_active": False,
    "anomaly_score": 0.0
}
_metrics_lock       = threading.Lock()



def video_reader_thread(inference: AsyncInferenceThread):
   
    global _video_source

    cap        = None
    src        = None
    frame_delay = 1.0 / 25.0

    while True:
        _source_changed.clear()

        with _video_source_lock:
            src = _video_source

        if src is None:
            # No video uploaded yet, just wait.
            time.sleep(1)
            continue

        # Open the new source
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)

        if not cap.isOpened():
            print(f"[Reader] Cannot open source: {src}. Retrying in 2s…")
            time.sleep(2)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 25.0
        print(f"[Reader] Opened: {src}  |  FPS={fps:.1f}  |  delay={frame_delay*1000:.1f}ms")

        while not _source_changed.is_set():
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                # End of file → loop back to frame 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Push to inference (non-blocking: drop if full to stay real-time)
            inference.push_frame(frame)

            # Sleep remaining time to match native FPS
            elapsed = time.time() - t_start
            sleep_t = frame_delay - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    if cap:
        cap.release()



# THREAD 2 — Result Collector  (pulls outputs from inference queue)

def result_collector_thread(inference: AsyncInferenceThread):
   
    global _latest_jpeg, _latest_metrics

    while True:
        annotated_frame, metrics = inference.get_output()
        if annotated_frame is None:
            time.sleep(0.005)    
            continue

        # Encode to JPEG (quality 85 is good quality / small size balance)
        ret, buffer = cv2.imencode(".jpg", annotated_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

       
        rgb = metrics.get("color", (0, 255, 0))
        hex_color = "#{:02x}{:02x}{:02x}".format(rgb[2], rgb[1], rgb[0])

        with _jpeg_lock:
            _latest_jpeg = buffer.tobytes()

        with _metrics_lock:
            _latest_metrics = {
                "count"        : int(metrics.get("count", 0)),
                "density"      : round(float(metrics.get("density", 0.0)), 2),
                "motion"       : round(float(metrics.get("motion", 0.0)), 2),
                "risk"         : metrics.get("risk", "SAFE"),
                "color"        : hex_color,
                "csrnet_active": bool(metrics.get("csrnet_active", False)),
                "anomaly_score": round(float(metrics.get("anomaly_score", 0.0)), 2)
            }

        _jpeg_ready.set()


# ═══════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════
def _mjpeg_generator():
    """Generator that yields the latest JPEG as an MJPEG stream with Safari-friendly limits."""
    while True:
        # Wait up to 1 s for a new frame (prevents CPU spinning)
        _jpeg_ready.wait(timeout=1.0)
        _jpeg_ready.clear()

        with _jpeg_lock:
            frame_bytes = _latest_jpeg

        if frame_bytes is None:
            # Yield a 1x1 black pixel or just yield nothing to keep the connection alive
            # Yielding a blank frame prevents Broken Pipe errors when browser connects before video loads
            # A tiny 1x1 black JPEG:
            blank_jpg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x00\xff\xd9'
            frame_bytes = blank_jpg

        # Safari strictly requires Content-Length for each multipart frame to render MJPEG
        header = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
        )
        yield header + frame_bytes + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    response = Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    # Prevent aggressive browser caching (crucial for Safari)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route("/api/metrics")
def api_metrics():
    with _metrics_lock:
        response = jsonify(_latest_metrics)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        return response


@app.route("/api/set_source", methods=["POST"])
def api_set_source():
    global _video_source

    # ── File upload ──────────────────────────────────────────
    if "file" in request.files:
        f = request.files["file"]
        if f.filename:
            fname = secure_filename(f.filename)
            fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            f.save(fpath)
            with _video_source_lock:
                _video_source = fpath
            
            # --- Reset Logic: Wipe old data before starting new video ---
            with _metrics_lock:
                _latest_metrics.update({"count": 0, "density": 0.0, "motion": 0.0, "risk": "SAFE", "anomaly_score": 0.0})
            with _jpeg_lock:
                global _latest_jpeg
                _latest_jpeg = None
            inference.reset_inference()
            
            _source_changed.set()   # tell reader thread to reload
            return jsonify({"status": "success", "message": f"Loaded: {fname}"})

    # ── JSON path / webcam id ─────────────────────────────────
    data = request.get_json(silent=True)
    if data and "source" in data:
        with _video_source_lock:
            _video_source = data["source"]
            
        # --- Reset Logic: Wipe old data before starting new video ---
        with _metrics_lock:
            _latest_metrics.update({"count": 0, "density": 0.0, "motion": 0.0, "risk": "SAFE", "anomaly_score": 0.0})
        with _jpeg_lock:
            _latest_jpeg = None
        inference.reset_inference()
        
        _source_changed.set()
        return jsonify({"status": "success", "message": f"Source: {data['source']}"})

    return jsonify({"status": "error", "message": "No valid source"}), 400


# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Initialising AI Inference Engine …")
    inference = AsyncInferenceThread()
    inference.start()

    # Start background threads (daemon = auto-killed when Flask exits)
    t_reader = threading.Thread(target=video_reader_thread,
                                args=(inference,), daemon=True, name="VideoReader")
    t_reader.start()

    t_collector = threading.Thread(target=result_collector_thread,
                                   args=(inference,), daemon=True, name="ResultCollector")
    t_collector.start()

    print("[INFO] Web Dashboard → http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, threaded=True)
