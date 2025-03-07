import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------
# Configuration
# ----------------------------
img_height = 150
img_width = 150
class_names = ["paper", "rock", "scissors"]

# Load the trained model (change filename if using the native Keras format)
model = load_model("rps_model.h5")

# ----------------------------
# Hand Detection using HSV Thresholding
# ----------------------------
def detect_hand(frame):
    """
    Detects the hand region using a simple skin color segmentation.
    Returns the bounding box (x, y, w, h) if a hand is detected, otherwise None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for skin detection (tweak these values as needed)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour corresponds to the hand
        max_contour = max(contours, key=cv2.contourArea)
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(max_contour)
        return (x, y, w, h)
    else:
        return None

# ----------------------------
# Prediction on the Cropped Hand Region
# ----------------------------
def predict_hand(frame, bbox):
    """
    Crops the hand region from the frame using the provided bounding box,
    preprocesses it, and returns the predicted class and the bot's winning move.
    """
    x, y, w, h = bbox
    hand_roi = frame[y:y+h, x:x+w]
    
    # Ensure the ROI is sufficiently large
    if hand_roi.size == 0 or hand_roi.shape[0] < 20 or hand_roi.shape[1] < 20:
        return None, None

    # Convert BGR (OpenCV default) to RGB and resize to training dimensions
    roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (img_width, img_height))
    
    # Convert to float32. Since the model includes a rescaling layer, no manual normalization is needed.
    roi_array = np.array(roi_resized, dtype=np.float32)
    roi_array = np.expand_dims(roi_array, axis=0)
    
    predictions = model.predict(roi_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    
    # Define the winning move mapping
    winning_moves = {
        "rock": "paper",       # paper beats rock
        "paper": "scissors",   # scissors beat paper
        "scissors": "rock"     # rock beats scissors
    }
    winning_move = winning_moves.get(predicted_class, "Unknown")
    
    return predicted_class, winning_move

# ----------------------------
# Main Loop: Real-Time Video Prediction with Bounding Box
# ----------------------------
def main():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hand and get bounding box
        bbox = detect_hand(frame)
        if bbox is not None:
            x, y, w, h = bbox
            # Draw the bounding box around the hand
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get predictions from the hand region
            predicted_class, winning_move = predict_hand(frame, bbox)
            if predicted_class is not None:
                # Overlay prediction text near the bounding box
                text = f"Predicted: {predicted_class} | Bot: {winning_move}"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow("Real-Time Hand Gesture Prediction", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()