import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Load the trained model
model = tf.keras.models.load_model("Model/hand_gesture_model.h5")

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # Detect one hand at a time for accuracy

# Load gesture labels
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
               "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        break

    hands, img = detector.findHands(img, draw=True)  # Detect hand(s)

    if hands:
        hand = hands[0]  # Take the first detected hand
        x, y, w, h = hand['bbox']  # Get bounding box

        # Ensure bounding box is within image dimensions
        img_height, img_width, _ = img.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)

        # Check if bounding box is valid
        if w > 0 and h > 0:
            # Extract hand region
            hand_img = img[y:y + h, x:x + w]
            hand_img = cv2.resize(hand_img, (128, 128))  # Resize for model input
            hand_img = hand_img / 255.0  # Normalize
            hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(hand_img)
            index = np.argmax(prediction)  # Get class index

            # Print detected gesture
            print(f"Detected Gesture: {class_names[index]}")

            # Display label on screen
            cv2.putText(img, class_names[index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            print("Invalid bounding box. Skipping prediction.")
    else:
        print("No hand detected.")

    cv2.imshow("Hand Gesture Recognition", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()