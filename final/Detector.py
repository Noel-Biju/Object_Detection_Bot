import cv2
import torch
from ultralytics import YOLO
import sys

try:
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for faster inference
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

try:
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    # Add pause state variable
    is_paused = False

    while True:
        # Only capture new frame if not paused
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 model on the frame
            results = model(frame)
            
            # Draw detection results on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = model.names[int(box.cls[0])]
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)
        
        # Add pause status text only when paused
        if is_paused:
            cv2.putText(frame, "PAUSED", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        
        # Display output
        cv2.imshow("YOLOv8 Object Detection", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):  # 'f' to quit
            break
        elif key == ord('p'):  # 'p' to pause/unpause
            is_paused = not is_paused

except KeyboardInterrupt:
    print("\nDetection stopped by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release resources
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()