#Write Python program to implement CNN object detection. Discuss numerous performance
#evaluation metrics for evaluating the object detecting algorithms' performance

import cv2
import numpy as np

def simple_object_detection(image_path, confidence_threshold=0.5):
    """
    Perform object detection using a pre-trained MobileNet SSD
    
    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence for detection
    """
    # Load pre-trained model
    net = cv2.dnn.readNetFromCaffe(
        'MobileNetSSD_deploy.prototxt.txt',
        'MobileNetSSD_deploy.caffemodel')
    
    # Class labels
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
               "train", "tvmonitor"]
               
    # Load image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                 0.007843, (300, 300), 127.5)
    
    # Forward pass
    net.setInput(blob)
    detections = net.forward()
    
    # Process results
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            
            # Get coordinates
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw bounding box
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "test_image.JPEG"
    
    # Run detection
    result = simple_object_detection(image_path)
    
    # Display result
    cv2.imshow("Object Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()