import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, target_size=(640, 480), apply_clahe=False):
        """
        Initializes the Video Processor module.
        target_size: Resolution to resize frames to (width, height)
        apply_clahe: If True, applies Contrast Limited Adaptive Histogram Equalization for low-light enhancement.
        """
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if apply_clahe else None

    def process_frame(self, frame):
        """
        Takes a raw BGR frame, resizes it, and optionally enhances contrast.
        Returns the processed frame.
        """
        # Resize frame to standard target size to reduce computation downstream
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

        if self.apply_clahe:
            # Convert to LAB color space to apply CLAHE to the Luminance channel only
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            final_frame = resized
            
        return final_frame

if __name__ == "__main__":
    # Test the module independently
    print("[INFO] Starting Preprocessing Module Test...")
    processor = VideoProcessor(target_size=(640, 480), apply_clahe=False)
    # Try opening the local video file
    cap = cv2.VideoCapture("videos/test.mp4")
    
    if not cap.isOpened():
        print("[ERROR] Could not open videos/test.mp4.")
        exit()
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam or video file.")
        exit()
        
    while True:
        ret, frame = cap.read()
        if not ret:
            # Restart video to loop infinitely for recording
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        processed_frame = processor.process_frame(frame)
        
        # Display side-by-side (Resize original to match height for appending)
        h, w = processed_frame.shape[:2]
        orig_resized = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
        
        cv2.imshow("Module 1 - Preprocessing (Processed)", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
