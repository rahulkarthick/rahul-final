from ultralytics import YOLO

class YOLODetector:

    def __init__(self, model_path):
        # Load trained YOLO model
        self.model = YOLO(model_path)

    def detect(self, frame):

        # Run detection
        results = self.model(frame)

        boxes = results[0].boxes
        person_count = 0

        # Count only persons (class id = 0)
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                person_count += 1

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        return annotated_frame, person_count