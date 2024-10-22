import streamlit as st
import joblib
import pandas as pd
import base64

# Load models
sclf1 = joblib.load('models/stacking_classifier_model1.pkl')
sclf2 = joblib.load('models/stacking_classifier_model2.pkl')
sclf3 = joblib.load('models/stacking_classifier_model3.pkl')
sclf4 = joblib.load('models/stacking_classifier_model4.pkl')

# Set page title and layout
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Load and encode the image as base64
def load_image(image_file):
    with open(image_file, "rb") as img:
        return base64.b64encode(img.read()).decode()

# Set the path for your local image
background_image = load_image("watermark.png")  # Local image file
background_style = f"url(data:image/png;base64,{background_image})"

# CSS to set the background image
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: {background_style};
        background-size: cover;
        background-position: center;
    }}
    .sidebar .sidebar-content {{
        background: rgba(255, 255, 255, 0.7);  /* Optional: sidebar background */
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Heart Disease Prediction App")
st.write("Please fill in the details below to get a prediction.")

# Create input fields in columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age:", min_value=0)
    sex = st.selectbox("Sex:", ["Male (1)", "Female (0)"])
    chest_pain_type = st.selectbox("Chest Pain Type:", [0, 1, 2, 3])
    resting_bp = st.number_input("Resting Blood Pressure:", min_value=0)
    cholesterol = st.number_input("Cholesterol:", min_value=0)

with col2:
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar (1=True, 0=False):", [0, 1])
    resting_ecg = st.selectbox("Resting ECG:", [0, 1, 2])
    max_heart_rate = st.number_input("Max Heart Rate:", min_value=0)
    exercise_angina = st.selectbox("Exercise Angina (1=Yes, 0=No):", [0, 1])
    oldpeak = st.number_input("Oldpeak:", format="%.1f")
    st_slope = st.selectbox("ST Slope:", [0, 1, 2])

# Choose model
model_choice = st.selectbox("Choose a Model:", ["Stacking Model 1", "Stacking Model 2", "Stacking Model 3", "Stacking Model 4"])

# Prediction button
if st.button("Predict"):
    # Map model choice to actual model
    model_mapping = {
        "Stacking Model 1": sclf1,
        "Stacking Model 2": sclf2,
        "Stacking Model 3": sclf3,
        "Stacking Model 4": sclf4
    }
    model = model_mapping[model_choice]
    
    # Prepare input data
    input_features = {
        'age': age,
        'sex': 1 if sex == "Male (1)" else 0,
        'chest_pain_type': chest_pain_type,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'resting_ecg': resting_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_angina': exercise_angina,
        'oldpeak': oldpeak,
        'st_slope': st_slope
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_features])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display result
    st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
