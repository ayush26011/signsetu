import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

# Text to speech
engine = pyttsx3.init()

last_spoken = ""
sentence = []

# Stable gesture detection variables
stable_gesture = ""
gesture_count = 0
HOLD_FRAMES = 10

# Load model & encoder
model = joblib.load("sign_model.pkl")
le = joblib.load("label_encoder.pkl")

EXPECTED_FEATURES = model.n_features_in_

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Camera start
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

print("📷 Camera ON | Q = Quit | C = Clear Sentence | S = Speak Sentence")

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Frame capture failed")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    prediction_text = "No Hand"

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        features = []

        base = hand.landmark[0]

        for lm in hand.landmark[:21]:

            features.append(lm.x - base.x)
            features.append(lm.y - base.y)

        features = np.array(features, dtype=np.float32)

        # Padding protection
        if features.shape[0] < EXPECTED_FEATURES:

            features = np.pad(
                features,
                (0, EXPECTED_FEATURES - features.shape[0]),
                mode="constant"
            )

        features = features.reshape(1, -1)

        try:

            pred_idx = model.predict(features)[0]

            prediction_text = le.inverse_transform([pred_idx])[0]

        except:
            prediction_text = "Prediction Error"

        # -------- Stable Gesture Detection --------

        if prediction_text == stable_gesture:

            gesture_count += 1

        else:

            stable_gesture = prediction_text
            gesture_count = 0

        if gesture_count == HOLD_FRAMES and prediction_text != last_spoken:

            sentence.append(prediction_text)

            if len(sentence) > 6:
                sentence.pop(0)

            # Speak word
            engine.say(prediction_text)
            engine.runAndWait()

            last_spoken = prediction_text

    # Show prediction
    cv2.putText(
        frame,
        f"Prediction: {prediction_text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Show sentence
    cv2.putText(
        frame,
        f"Sentence: {' '.join(sentence)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow("SignSetu Live Test", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Clear sentence
    if key == ord("c"):

        sentence.clear()

        last_spoken = ""

        stable_gesture = ""

        gesture_count = 0

        print("Sentence Cleared")

    # Speak full sentence
    if key == ord("s"):

        if sentence:

            full_sentence = " ".join(sentence)

            engine.say(full_sentence)

            engine.runAndWait()

cap.release()

cv2.destroyAllWindows()