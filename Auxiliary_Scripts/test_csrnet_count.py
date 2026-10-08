import cv2
from prediction_module import CSRNetEstimator, YOLOv8Detector

csrnet = CSRNetEstimator()
yolo = YOLOv8Detector()

cap = cv2.VideoCapture("videos/test.mp4")
ret, frame = cap.read()
if ret:
    raw_map, csrnet_count = csrnet.predict(frame)
    bboxes, yolo_count, yolo_annotated = yolo.detect(frame)
    print(f"RAW CSRNET COUNT: {csrnet_count}")
    print(f"RAW YOLO COUNT: {yolo_count}")
