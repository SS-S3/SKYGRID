import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image  # <-- Import Image from PIL
import numpy as np

# Load the pre-trained DETR model and processor (image processor)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Set the model to evaluation mode
model.eval()

# Use CUDA if available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the video capture
cap = cv2.VideoCapture('/Users/soumyashekhar/Desktop/test.mp4')  ##'rtsp://your_drone_ip_address:your_drone_port'

# Get the FPS (frames per second) of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Video FPS: {fps}")

frame_interval = fps  # Skip frames to process one frame per second

# Track the frame count
frame_count = 0

while True:
    # Capture a frame from the livestream
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # Process only one frame per second
    if frame_count % frame_interval == 0:
        # Convert the frame to a PIL image for DETR
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)  # Convert numpy array to PIL Image

        # Preprocess the image and pass it through the model
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        # Get predictions from the model
        with torch.no_grad():  # Disable gradient computation
            outputs = model(**inputs)

        # Get the predicted bounding boxes, labels, and scores
        target_sizes = torch.tensor([pil_image.size[::-1]])  # Fix this part, ensure it's (height, width)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        human_count = 0
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 1:  # 1 corresponds to 'person' in COCO dataset
                # Extract bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.tolist()

                # Draw bounding box around detected person
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                human_count += 1
        
        # Print the number of humans detected in the terminal
        print(f"Frame {frame_count}: Humans Detected: {human_count}")
        
        # Optionally display the count on the frame (video window)
        cv2.putText(frame, f'Humans Detected: {human_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the output
    cv2.imshow('Human Detection with DETR', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captureœ
cap.release()
cv2.destroyAllWindows()
