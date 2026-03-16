from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("sign_model.pkl")
le = joblib.load("label_encoder.pkl")

EXPECTED_FEATURES = model.n_features_in_

CONFIDENCE_THRESHOLD = 0.80

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

@app.route('/')
def home():
    return render_template("index.html")


# Prediction API
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