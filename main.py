import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2

# ----------------------------
# Configuration
# ----------------------------
data_dir = "dataset"  # Dataset folder with subfolders 'rock', 'paper', 'scissors'
batch_size = 32
img_height = 150
img_width = 150
epochs = 10  # Adjust epochs as needed

# ----------------------------
# Define Preprocessing Functions
# ----------------------------
def crop_and_blur(image_np):
    """
    Process an image (as a NumPy array) by detecting the hand region using HSV skin segmentation,
    cropping to the hand, and blending the hand region with a blurred background.
    Returns a resized image (img_width x img_height) as a NumPy array in RGB format.
    """
    # image_np is assumed to be in RGB (dtype=uint8)
    # Convert to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Convert BGR image to HSV color space
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for skin color (tweak these values if necessary)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clean up the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour corresponds to the hand
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        if w > 20 and h > 20:
            # Crop the hand region
            # Optionally, you could use only the cropped region; here we blend it with a blurred background.
            cropped = image_bgr[y:y+h, x:x+w]
            # Create a blurred version of the entire image
            blurred = cv2.GaussianBlur(image_bgr, (21, 21), 0)
            # Create a mask for the hand region
            hand_mask = np.zeros_like(image_bgr)
            cv2.drawContours(hand_mask, [max_contour], -1, (255, 255, 255), -1)
            # Combine: keep the hand region in focus, and the rest blurred.
            combined = np.where(hand_mask == np.array([255, 255, 255]), image_bgr, blurred)
            # Crop the combined image to the bounding box of the hand
            processed = combined[y:y+h, x:x+w]
            # Resize to the target dimensions
            processed_resized = cv2.resize(processed, (img_width, img_height))
            # Convert back to RGB
            processed_rgb = cv2.cvtColor(processed_resized, cv2.COLOR_BGR2RGB)
            return processed_rgb

    # If no hand is detected or region is too small, simply resize the original image.
    image_resized = cv2.resize(image_bgr, (img_width, img_height))
    processed_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    return processed_rgb

def preprocess_image(image):
    """
    TensorFlow wrapper to apply the crop_and_blur function.
    Expects image as a tensor and returns a tensor.
    """
    # Use tf.py_function to wrap the NumPy/OpenCV processing
    processed = tf.py_function(func=crop_and_blur, inp=[image], Tout=tf.uint8)
    # Set the shape information manually
    processed.set_shape([img_height, img_width, 3])
    return processed

def preprocess_batch(images, labels):
    """
    Applies the preprocess_image function to each image in a batch.
    """
    processed_images = tf.map_fn(preprocess_image, images, fn_output_signature=tf.uint8)
    return processed_images, labels

# ----------------------------
# Load Raw Dataset with Validation Split
# ----------------------------
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Save the class names before mapping (so they are preserved)
class_names = raw_train_ds.class_names
print("Classes found:", class_names)

# Apply our preprocessing to crop hands and blur the background
train_ds = raw_train_ds.map(preprocess_batch)
val_ds = raw_val_ds.map(preprocess_batch)

# ----------------------------
# Data Augmentation and Normalization (Optional)
# ----------------------------
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
normalization_layer = layers.Rescaling(1.0 / 255)

# ----------------------------
# Build the CNN Model
# ----------------------------
num_classes = len(class_names)

model = models.Sequential([
    data_augmentation,
    normalization_layer,
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------------
# Train the Model
# ----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Evaluate on the validation set
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc * 100:.2f}%")

# Save the trained model in native Keras format
model.save("my_model.keras")

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(image_path, model, class_names):
    """
    Load an image, preprocess it by cropping the hand and blurring the background,
    predict its class, and output the winning move.
    
    For the game:
      - paper beats rock,
      - scissors beat paper,
      - rock beats scissors.
    """
    from tensorflow.keras.preprocessing import image as keras_image

    img = keras_image.load_img(image_path, target_size=(img_height, img_width))
    img_array = keras_image.img_to_array(img)
    # Preprocess the image using the same crop and blur routine
    img_processed = crop_and_blur(img_array.astype(np.uint8))
    img_processed = np.expand_dims(img_processed, axis=0)  # Create batch dimension

    predictions = model.predict(img_processed)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    print("Predicted move:", predicted_class)

    # Determine the winning move
    winning_moves = {
        "rock": "paper",       # paper beats rock
        "paper": "scissors",   # scissors beat paper
        "scissors": "rock"     # rock beats scissors
    }
    winning_move = winning_moves.get(predicted_class, "Unknown")
    print("Bot should play:", winning_move)
    return predicted_class, winning_move

if __name__ == "__main__":
    # For testing the prediction function, you can uncomment the following lines:
    # test_image_path = "path/to/your/test_image.jpg"
    # predict_image(test_image_path, model, class_names)
    pass