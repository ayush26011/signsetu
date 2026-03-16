from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("sign_model.pkl")
le = joblib.load("label_encoder.pkl")

EXPECTED_FEATURES=model.n_features_in_

mp_hands=mp.solutions.hands

hands=mp_hands.Hands(
static_image_mode=False,
max_num_hands=1,
min_detection_confidence=0.7,
min_tracking_confidence=0.7
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():

    try:

        if "image" not in request.files:
            return jsonify({"prediction":"No Image"})

        file=request.files["image"]

        img_bytes=file.read()

        npimg=np.frombuffer(img_bytes,np.uint8)

        frame=cv2.imdecode(npimg,cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"prediction":"Decode Error"})

        frame=cv2.flip(frame,1)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        result=hands.process(rgb)

        prediction="No Hand"

        if result.multi_hand_landmarks:

            hand=result.multi_hand_landmarks[0]

            features=[]
            base=hand.landmark[0]

            for lm in hand.landmark[:21]:

                features.append(lm.x-base.x)
                features.append(lm.y-base.y)

            features=np.array(features,dtype=np.float32)

            if features.shape[0] < EXPECTED_FEATURES:

                features=np.pad(
                features,
                (0,EXPECTED_FEATURES-features.shape[0]),
                mode="constant"
                )

            features=features.reshape(1,-1)

            pred=model.predict(features)[0]

            prediction=le.inverse_transform([pred])[0]

        return jsonify({"prediction":prediction})

    except Exception as e:

        print("Error:",str(e))

        return jsonify({"prediction":"Error"})


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)