import cv2
import torch
from torchvision.models import detection
import time

# Load the Faster R-CNN model
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Set up the video capture
cap = cv2.VideoCapture('/Users/soumyashekhar/Downloads/Untitled.mp4') ##'rtsp://your_drone_ip_address:your_drone_port'

while True:
    # Capture a frame from the livestream
    ret, frame = cap.read()
    
    
    if not ret:
        break
    
    # Convert the frame to a PyTorch tensor
    tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
    
    # Add a batch dimension
    tensor = tensor.unsqueeze(0)
    
    # Run Faster R-CNN on the frame
    with torch.no_grad():  # Disable gradient computation
        outputs = model(tensor)
    
    # Get the detected bounding boxes, class labels, and scores
    for output in outputs:
        scores = output['scores'].cpu().numpy()  # Convert to NumPy array for easier indexing
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()  # COCO dataset class IDs
        
        for i, score in enumerate(scores):
            if score > 0.5 and labels[i] == 1:  # 1 is the class ID for 'person' in COCO dataset
                # Get the bounding box coordinates
                x1, y1, x2, y2 = boxes[i].tolist()
                
                # Draw bounding boxes around the detected humans
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Display the output
    cv2.imshow('Human Detection', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()

