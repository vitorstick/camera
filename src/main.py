import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
FROM_EMAIL = os.getenv('FROM_EMAIL')
TO_EMAIL = os.getenv('TO_EMAIL')

def setup_video_capture():
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Add cv2.CAP_DSHOW for Windows
        if cap.isOpened():
            # Try to read a frame to make sure it's working
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Successfully connected to camera {camera_index}")
                return cap
            else:
                print(f"Camera {camera_index} opened but couldn't read frame")
                cap.release()
        else:
            print(f"Could not open camera {camera_index}")
    
    print("Error: Could not find any working webcam.")
    exit()

def load_yolo_model():
    model = YOLO('yolov8n.pt')  # Load the standard YOLOv8 model
    # List of bird-related classes in COCO dataset that YOLOv8 can detect
    bird_classes = [
        'bird', 'person'
    ]
    return model, bird_classes

def send_email_with_image(image_path, confidence, detected_class):
    try:
        # Read the image file
        with open(image_path, 'rb') as f:
            data = f.read()
        
        # Encode the image
        encoded_file = base64.b64encode(data).decode()

        # Create the email
        message = Mail(
            from_email=FROM_EMAIL,
            to_emails=TO_EMAIL,
            subject=f'{detected_class.capitalize()} Detected!',
            html_content=f'''
                <h2>{detected_class.capitalize()} Detection Alert!</h2>
                <p>A {detected_class} was detected with {confidence:.2%} confidence!</p>
                <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Image is attached.</p>
            '''
        )

        # Attach the image
        attachedFile = Attachment(
            FileContent(encoded_file),
            FileName(os.path.basename(image_path)),
            FileType('image/png'),
            Disposition('attachment')
        )
        message.attachment = attachedFile

        # Send the email
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        if response.status_code == 202:
            print("Email sent successfully!")
        
    except Exception as e:
        print(f"Warning - Email sending failed: {str(e)}")
        # Continue program execution even if email fails

def save_detected_image(frame, confidence, detected_class):
    try:
        # Create 'captures' directory if it doesn't exist
        if not os.path.exists('captures'):
            os.makedirs('captures')
        
        # Generate filename with timestamp and confidence
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'captures/{detected_class}_{timestamp}_{confidence:.2f}.png'
        
        # Save the image
        cv2.imwrite(filename, frame)
        print(f"{detected_class.capitalize()} image saved as: {filename}")
        
        # Send email with the image
        send_email_with_image(filename, confidence, detected_class)
    except Exception as e:
        print(f"Warning - Image saving failed: {str(e)}")
        # Continue program execution even if saving fails

def process_frames(cap):
    model, detected_classes = load_yolo_model()
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    print("Detection Started - Press 'q' to quit")
    print("Watching for detections...")
    
    # To prevent multiple saves of the same detection
    last_save_time = 0
    save_cooldown = 120  # 2 minutes between captures

    while True:
        try:
            if not cap.isOpened():
                print("Error: Camera disconnected. Attempting to reconnect...")
                cap = setup_video_capture()
                continue

            ret, frame1 = cap.read()
            if not ret or frame1 is None:
                print("Error reading frame. Retrying...")
                continue

            # Motion detection
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            
            # Object detection
            results = model(frame1, verbose=False)
            
            # Process detections
            current_time = datetime.now().timestamp()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name and confidence
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    conf = float(box.conf[0])
                    
                    # Only process if it's a detected class and confidence is above 0.7
                    if class_name in detected_classes and conf > 0.7:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Check if there's motion in the detected area
                        detected_roi = dilated[y1:y2, x1:x2]
                        if detected_roi.any():
                            # Draw rectangle and label with class name
                            label = f'{class_name}: {conf:.2f}'
                            cv2.putText(frame1, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Save image and send email if enough time has passed
                            if current_time - last_save_time >= save_cooldown:
                                save_detected_image(frame1.copy(), conf, class_name)  # Pass detected class
                                last_save_time = current_time
                                print(f"{class_name} detected with {conf:.2%} confidence!")

            # Show the frame
            cv2.imshow("Movement Detection", frame1)
            frame2 = frame1.copy()  # Use a copy to avoid reference issues

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Warning - Frame processing error: {str(e)}")
            continue  # Continue to next frame even if there's an error

    cap.release()
    cv2.destroyAllWindows()
    print("Detection Stopped")

if __name__ == "__main__":
    if not SENDGRID_API_KEY or not FROM_EMAIL or not TO_EMAIL:
        print("Error: Please set up your .env file with SENDGRID_API_KEY, FROM_EMAIL, and TO_EMAIL")
        exit()
    
    cap = setup_video_capture()
    process_frames(cap)