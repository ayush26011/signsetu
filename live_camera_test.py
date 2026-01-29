import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model & encoder
model = joblib.load("sign_model.pkl")
le = joblib.load("label_encoder.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Camera ON | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction_text = "NO HAND"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = []

        # WRIST as base
        base_x = hand.landmark[0].x
        base_y = hand.landmark[0].y

        for lm in hand.landmark[:21]:
            features.append(lm.x - base_x)
            features.append(lm.y - base_y)

        # 🔥 MUST MATCH TRAINING FEATURES (42)
        if len(features) == 42:
            features = np.array(features, dtype=np.float32).reshape(1, -1)

            pred_idx = model.predict(features)[0]
            prediction_text = le.inverse_transform([pred_idx])[0]

    cv2.putText(
        frame,
        f"Prediction: {prediction_text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("ISL Live Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
