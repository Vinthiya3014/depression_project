from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)

DATA_PATH = "student_mental_health_data (1).csv"
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()
categorical_cols = ['gender', 'family_support', 'outdoor_activity', 'counseling_aware']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("family_support classes:", label_encoders['family_support'].classes_)
X = df[['age', 'gender', 'study_hours', 'sleep_hours',
        'social_media_hours', 'family_support', 'activity_hours',
        'outdoor_activity', 'counseling_aware']]
y = df['depressed']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

os.makedirs('model', exist_ok=True)
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/form", methods=["GET"])
def form():
    with open('model/label_encoders.pkl', 'rb') as f:
        encs = pickle.load(f)
    family_support_options = encs['family_support'].classes_.tolist()
    gender_options          = encs['gender'].classes_.tolist()
    outdoor_options         = encs['outdoor_activity'].classes_.tolist()
    counseling_options      = encs['counseling_aware'].classes_.tolist()
    return render_template("form.html",
                           gender_options=gender_options,
                           family_support_options=family_support_options,
                           outdoor_options=outdoor_options,
                           counseling_options=counseling_options)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        with open('model/rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model/label_encoders.pkl', 'rb') as f:
            encs = pickle.load(f)

        def safe_transform(le, val):
            try:
                return le.transform([val])[0]
            except ValueError:
                return None
        age                = int(request.form["age"])
        study_hours        = float(request.form["study_hours"])
        sleep_hours        = float(request.form["sleep_hours"])
        social_media_hours = float(request.form["social_media_hours"])
        activity_hours     = float(request.form["activity_hours"])
        gender            = safe_transform(encs['gender'], request.form["gender"])
        family_support    = safe_transform(encs['family_support'], request.form["family_support"])
        outdoor_activity  = safe_transform(encs['outdoor_activity'], request.form["outdoor_activity"])
        counseling_aware  = safe_transform(encs['counseling_aware'], request.form["counseling_aware"])

        if None in [gender, family_support, outdoor_activity, counseling_aware]:
            return "Error: invalid category submitted.", 400
        features = np.array([[age, gender, study_hours, sleep_hours,
                              social_media_hours, family_support,
                              activity_hours, outdoor_activity,
                              counseling_aware]])
        feat_scaled = scaler.transform(features)
        pred = model.predict(feat_scaled)[0]
        proba = model.predict_proba(feat_scaled)[0][pred]

        result = "Depressed" if pred == 1 else "Not Depressed"
        feedback = []

        if sleep_hours < 6:
            feedback.append("Try to get at least 7–8 hours of sleep each night. Sleep is essential for emotional balance and academic performance.")
        elif sleep_hours > 10:
            feedback.append("Oversleeping can sometimes be a sign of mental distress or fatigue. Consider maintaining a regular sleep schedule.")

        if study_hours > 8:
            feedback.append("Studying too long without breaks may lead to burnout. Try using techniques like the Pomodoro method to manage time better.")
        elif study_hours < 2:
            feedback.append("Try to increase study hours gradually with structured goals to improve your academic confidence.")

        if social_media_hours > 4:
            feedback.append("Heavy social media use can increase stress and anxiety. Try limiting screen time and focus on offline activities.")
        elif social_media_hours < 1:
            feedback.append("Staying offline is great, but be sure you're not isolating yourself. Balance digital detox with social connection.")

        if family_support == 0:
            feedback.append("Lack of family support can feel isolating. Consider talking to a counselor or seeking support from friends or mentors.")

        if activity_hours < 1:
            feedback.append("Engaging in physical activity for even 30 minutes a day can boost mood and reduce stress.")
        elif activity_hours > 5:
            feedback.append("Be careful of overexertion. Balance your physical activity with rest and nutrition.")

        if outdoor_activity == 0:
            feedback.append("Spending time outdoors can improve mood and reduce anxiety. Even a short walk helps.")

        if counseling_aware == 0:
            feedback.append("Learn about the mental health support services your institution offers. Reaching out is a sign of strength, not weakness.")

        if not feedback:
            feedback.append("You're on a healthy track! Continue building good habits and keep checking in with yourself.")

        return render_template("result.html",
                               result=result,
                               confidence=round(proba*100, 2),
                               feedback=feedback)

    except Exception as e:
        return f"Error during prediction: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
