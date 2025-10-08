from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 model (use a larger model like yolov8m.pt for better accuracy)
model = YOLO('yolov8m.pt')

# Confidence threshold for better accuracy
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4  # Adjust for better performance

def detect_human(frame):
    # Perform inference using YOLO
    results = model(frame)
    
    # Get predictions for humans (class 0 is person) and apply NMS
    persons = []
    im_array = None
    for r in results:
        # Filter results based on class 0 (person) and confidence score
        person_boxes = r.boxes[(r.boxes.cls == 0) & (r.boxes.conf > CONFIDENCE_THRESHOLD)]  # Only keep person class with high confidence
        persons.extend(person_boxes)
        
        # Apply Non-Maximum Suppression (NMS) if necessary
        im_array = r.plot(conf=CONFIDENCE_THRESHOLD)  # Plot with confidence threshold applied
    
    return persons, im_array

def start_live_detection():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    prev_time = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Detect humans in frame
        persons, annotated_frame = detect_human(frame)
        
        # Add FPS text to frame
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        
        # Display number of persons detected
        cv2.putText(annotated_frame, f'Persons: {len(persons)}', (20, 40), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        
        # Display result
        cv2.imshow('Human Detection', annotated_frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_detection()
