import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import time

# Load the trained model
model = tf.keras.models.load_model("Model/ASL_2.h5")
print(f"Model input shape: {model.input_shape}")

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=3, detectionCon=0.8)  # Increased detection confidence

# Load gesture labels
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
               "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 'del', 'space', 'nothing']

# Improved prediction smoothing parameters
confidence_threshold = 0.6  # Slightly lowered from 0.7
prediction_history = []
prediction_window_size = 15  # Increased from 5 for more stability
frame_skip = 1  # Process every frame for better responsiveness
frame_count = 0

# Prediction stabilization variables
current_prediction = None
prediction_start_time = 0
prediction_stable_time = 1.0  # Display prediction for at least 1 second
last_display_time = 0
display_cooldown = 0.5  # Minimum time between prediction changes (seconds)

# Debug mode
debug_mode = True  # Set to False to disable debug info

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        cv2.imshow("Hand Gesture Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    current_time = time.time()
    hands, img = detector.findHands(img, draw=True)

    # Default status when no hands detected
    status_text = "No hand detected"
    confidence_text = ""
    text_color = (0, 0, 255)  # Red for no detection
    debug_info = []

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # More generous padding (30% instead of 20%)
        padding_x = int(w * 0.3)
        padding_y = int(h * 0.3)

        # Ensure bounding box with padding is within image dimensions
        img_height, img_width, _ = img.shape
        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(img_width, x + w + padding_x)
        y_end = min(img_height, y + h + padding_y)

        # Extract hand region with padding
        hand_img = img[y_start:y_end, x_start:x_end]

        if hand_img.size > 0:
            # Resize to match model input
            hand_img = cv2.resize(hand_img, (64, 64))

            # Apply preprocessing - normalize the image
            hand_img = hand_img / 255.0

            # Add batch dimension
            hand_img = np.expand_dims(hand_img, axis=0)

            # Make prediction
            prediction = model.predict(hand_img, verbose=0)
            index = np.argmax(prediction[0])
            confidence = prediction[0][index]

            # Add prediction to history
            prediction_history.append((index, confidence))
            if len(prediction_history) > prediction_window_size:
                prediction_history.pop(0)

            # Calculate weighted average prediction
            if prediction_history:
                # Count occurrences and weight by confidence
                class_scores = {}
                for idx, conf in prediction_history:
                    if idx not in class_scores:
                        class_scores[idx] = 0
                    class_scores[idx] += conf  # Weight by confidence

                # Find the class with highest weighted score
                smooth_index = max(class_scores.items(), key=lambda x: x[1])[0]
                smooth_confidence = class_scores[smooth_index] / sum(p[1] for p in prediction_history)

                # Add debug info
                if debug_mode:
                    # Show top 3 predictions
                    top_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    for cls_idx, score in top_classes:
                        norm_score = score / sum(p[1] for p in prediction_history)
                        debug_info.append(f"{class_names[cls_idx]}: {norm_score * 100:.1f}%")

                # Update prediction only if confidence exceeds threshold and enough time has passed
                if smooth_confidence > confidence_threshold:
                    # If this is a new prediction or enough time has passed
                    if current_prediction != smooth_index and current_time - last_display_time > display_cooldown:
                        current_prediction = smooth_index
                        prediction_start_time = current_time
                        last_display_time = current_time

                    # Use the stable prediction
                    status_text = class_names[current_prediction]
                    confidence_text = f"{smooth_confidence * 100:.1f}%"
                    text_color = (0, 255, 0)  # Green for confident prediction
                else:
                    status_text = "Uncertain"
                    confidence_text = f"{smooth_confidence * 100:.1f}%"
                    text_color = (0, 255, 255)  # Yellow for uncertain prediction
        else:
            status_text = "Hand too close to edge"

    # Display prediction on screen
    cv2.rectangle(img, (10, 10), (300, 150 if debug_mode else 70), (0, 0, 0), -1)  # Black background for text
    cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(img, confidence_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    # Display debug info
    if debug_mode and debug_info:
        for i, info in enumerate(debug_info):
            cv2.putText(img, info, (20, 90 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the hand bounding box
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()