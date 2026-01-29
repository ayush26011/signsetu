import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

label = input("Enter sign label: ")

data = []

print("Show hand, press S to save frame, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        base = hand.landmark[0]

        features = []
        for lm in hand.landmark[:21]:
            features.append(lm.x - base.x)
            features.append(lm.y - base.y)

        if len(features) == 42:
            cv2.putText(frame, "Hand Detected",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and result.multi_hand_landmarks:
        data.append(features + [label])
        print("Saved")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv("sign_data.csv", mode="a", header=False, index=False)

print("Data saved")
