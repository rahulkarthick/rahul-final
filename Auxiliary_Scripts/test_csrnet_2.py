import cv2
import glob
from prediction_module import CSRNetEstimator, YOLOv8Detector

csrnet = CSRNetEstimator()
yolo = YOLOv8Detector()

vids = glob.glob("videos/*.mp4")
target = None
for v in vids:
    if "6574285" in v:
        target = v
        break

if target:
    print(f"Testing {target}")
    cap = cv2.VideoCapture(target)
    ret, frame = cap.read()
    if ret:
        raw_map, csrnet_count = csrnet.predict(frame)
        bboxes, yolo_count, _ = yolo.detect(frame)
        print(f"RAW CSRNET COUNT: {csrnet_count}")
        print(f"RAW YOLO COUNT: {yolo_count}")
else:
    print("Video not found")
