from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import Counter

app = Flask(__name__)
CORS(app)
# Load model
model = joblib.load("sign_model.pkl")
le = joblib.load("label_encoder.pkl")

EXPECTED_FEATURES = model.n_features_in_

sentence = []
last_prediction = ""

# stability variables
prediction_buffer = []
BUFFER_SIZE = 12
COOLDOWN_FRAMES = 20
cooldown = 0

CONFIDENCE_THRESHOLD = 0.80

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)


def generate_frames():

    global sentence, last_prediction
    global prediction_buffer, cooldown

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        prediction = "No Hand"
        confidence = 0

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

            if features.shape[0] < EXPECTED_FEATURES:
                features = np.pad(
                    features,
                    (0, EXPECTED_FEATURES - features.shape[0]),
                    mode="constant"
                )

            features = features.reshape(1, -1)

            probs = model.predict_proba(features)

            confidence = np.max(probs)

            pred = np.argmax(probs)

            prediction = le.inverse_transform([pred])[0]

            # Confidence filter
            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_buffer.append(prediction)

                if len(prediction_buffer) > BUFFER_SIZE:
                    prediction_buffer.pop(0)

                if len(prediction_buffer) == BUFFER_SIZE:

                    most_common = Counter(prediction_buffer).most_common(1)[0][0]

                    if cooldown == 0 and most_common != last_prediction:

                        sentence.append(most_common)

                        if len(sentence) > 6:
                            sentence.pop(0)

                        last_prediction = most_common

                        cooldown = COOLDOWN_FRAMES

            else:
                prediction = "Uncertain"

        # cooldown reduce
        if cooldown > 0:
            cooldown -= 1

        # display prediction
        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        # confidence
        cv2.putText(
            frame,
            f"Confidence: {round(confidence*100,1)}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        # sentence
        cv2.putText(
            frame,
            "Sentence: " + " ".join(sentence),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sentence')
def get_sentence():
    return jsonify({"sentence": " ".join(sentence)})


@app.route('/clear')
def clear_sentence():
    global sentence, last_prediction, prediction_buffer
    sentence = []
    last_prediction = ""
    prediction_buffer = []
    return jsonify({"status": "cleared"})


# ⭐ NEW API ROUTE FOR FRONTEND
@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img_bytes = file.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = "No Hand"
    confidence = 0

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        features = []
        base = hand.landmark[0]

        for lm in hand.landmark[:21]:
            features.append(lm.x - base.x)
            features.append(lm.y - base.y)

        features = np.array(features, dtype=np.float32)

        if features.shape[0] < EXPECTED_FEATURES:
            features = np.pad(
                features,
                (0, EXPECTED_FEATURES - features.shape[0]),
                mode="constant"
            )

        features = features.reshape(1, -1)

        probs = model.predict_proba(features)

        confidence = np.max(probs)

        pred = np.argmax(probs)

        prediction = le.inverse_transform([pred])[0]

        if confidence < CONFIDENCE_THRESHOLD:
            prediction = "Uncertain"

    return jsonify({
        "prediction": prediction,
        "confidence": float(confidence)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)