import cv2
import torch
from ultralytics import YOLO
from config import MODEL_PATH, VEHICLE_CLASSES
import argparse

class CarDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)

    def detect_cars(self, frame):
        results = self.model(frame)
        detected_cars = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = self.model.names[cls]
                if class_name in VEHICLE_CLASSES:
                    detected_cars.append((x1, y1, x2, y2, conf, class_name))

        return detected_cars

    def draw_boxes(self, frame, cars):
        for car in cars:
            x1, y1, x2, y2, conf, cls = car
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def run_detection(self, source):
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error opening video source {source}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cars = self.detect_cars(frame)
            frame_with_boxes = self.draw_boxes(frame, cars)

            cv2.imshow("Car Detection", frame_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Car Detection using YOLO")
    parser.add_argument("--source", type=str, default="0", help="Path to video file or camera index (default: 0 for webcam)")
    args = parser.parse_args()

    detector = CarDetector()
    
    # If source is not a number, treat it as a file path
    if not args.source.isdigit():
        source = args.source
    else:
        source = int(args.source)
    
    detector.run_detection(source)

if __name__ == "__main__":
    main()