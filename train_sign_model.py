import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("sign_data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(str)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0).values.astype(np.float32)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test) * 100)

joblib.dump(model, "sign_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("MODEL SAVED")
