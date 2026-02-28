import cv2
from detection.yolo_detector import YOLODetector

# Load YOLO detector
detector = YOLODetector("models/yolo/best.pt")

video_path = "videos/test.mp4"
cap = cv2.VideoCapture(video_path)

print("YOLO Detection Started")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detection
    output_frame, count = detector.detect(frame)

    # Show count
    cv2.putText(
        output_frame,
        f"Person Count: {count}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("YOLO Crowd Detection", output_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()