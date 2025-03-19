import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

def setup_video_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    return cap

def load_yolo_model():
    model = YOLO('yolov8n.pt')  # Load the standard YOLOv8 model
    # List of bird-related classes in COCO dataset that YOLOv8 can detect
    bird_classes = ['bird']
    return model, bird_classes

def save_bird_image(frame, confidence):
    # Create 'bird_captures' directory if it doesn't exist
    if not os.path.exists('bird_captures'):
        os.makedirs('bird_captures')
    
    # Generate filename with timestamp and confidence
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'bird_captures/bird_{timestamp}_{confidence:.2f}.png'
    
    # Save the image
    cv2.imwrite(filename, frame)
    print(f"Bird image saved as: {filename}")

def process_frames(cap):
    model, bird_classes = load_yolo_model()
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    print("Bird Detection Started - Press 'q' to quit")
    print("Watching for birds...")
    
    # To prevent multiple saves of the same bird
    last_save_time = 0
    save_cooldown = 2  # Seconds between saves

    while cap.isOpened():
        # Motion detection
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        
        # Object detection
        results = model(frame1, verbose=False)  # Disable verbose output
        
        # Process detections
        bird_detected = False
        current_time = datetime.now().timestamp()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                cls = int(box.cls[0])
                class_name = result.names[cls]
                conf = float(box.conf[0])
                
                # Only process if it's a bird and confidence is above 0.5
                if class_name in bird_classes and conf > 0.5:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Check if there's motion in the bird area
                    bird_roi = dilated[y1:y2, x1:x2]
                    if bird_roi.any():
                        bird_detected = True
                        # Draw rectangle and label
                        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Bird: {conf:.2f}'
                        cv2.putText(frame1, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save image if enough time has passed since last save
                        if current_time - last_save_time >= save_cooldown:
                            save_bird_image(frame1, conf)
                            last_save_time = current_time
                            print(f"Bird detected with {conf:.2%} confidence!")

        # Show the frame
        cv2.imshow("Bird Movement Detection", frame1)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Bird Detection Stopped")

if __name__ == "__main__":
    cap = setup_video_capture()
    process_frames(cap)