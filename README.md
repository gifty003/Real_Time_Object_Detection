# Real_Time_Object_Detection
## NAME   : GIFTSON RAJARATHINAM N
## REG NO : 212222233002
## DEPT   : AI - DS

## AIM :
The real time object detection Using Web Camera.

## Software Required :
Anaconda - Python 3.7

## Algorithm :
### Step 1: Install OpenCV
### Step 2: Download YOLOv4 files
        Download yolov4.weights and yolov4.cfg from the YOLO repository.
        Download the coco.names file for COCO class labels.
### Step 3: Load YOLOv4 model
### Step 4: Load COCO class labels
### Step 5: Set up YOLO output layers
### Step 6: Capture video from webcam
### Step 7: Process each video frame
### Step 8: Detect objects in the frame
### Step 9: Apply Non-Max Suppression
### Step 10: Draw bounding boxes and labels
### Step 11: Display the output
### Step 12: Exit the loop
### Step 13: Release resources

## program :

```
import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO output
    outputs = net.forward(output_layers)
    
    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

```

### Output :
![Screenshot 2024-09-24 105033](https://github.com/user-attachments/assets/816d3658-fd4c-4a85-a776-a05d845ee65b)

## Result :
Thus the object in real time object is detected.
