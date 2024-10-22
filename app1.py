from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd  # Import pandas to create DataFrame

# Load all models
sclf1 = joblib.load('models/stacking_classifier_model1.pkl')
sclf2 = joblib.load('models/stacking_classifier_model2.pkl')
sclf3 = joblib.load('models/stacking_classifier_model3.pkl')
sclf4 = joblib.load('models/stacking_classifier_model4.pkl')

app = Flask(__name__)

# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in a templates folder

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print(f"Received data: {data}")  # Log the received data
        
        # Extract user inputs from the request data
        input_features = {
            'age': data['age'],
            'sex': data['sex'],
            'chest_pain_type': data['chest_pain_type'],
            'resting_bp': data['resting_bp'],  # Correct key name
            'cholesterol': data['cholesterol'],
            'fasting_blood_sugar': data['fasting_blood_sugar'],
            'resting_ecg': data['resting_ecg'],
            'max_heart_rate': data['max_heart_rate'],
            'exercise_angina': data['exercise_angina'],
            'oldpeak': data['oldpeak'],
            'st_slope': data['st_slope']
        }

        # Convert the input features to a DataFrame
        input_df = pd.DataFrame([input_features])
        print(f"Input DataFrame: {input_df}")  # Log the DataFrame

        model_choice = data['model']
        if model_choice == 'sclf1':
            model = sclf1
        elif model_choice == 'sclf2':
            model = sclf2
        elif model_choice == 'sclf3':
            model = sclf3
        elif model_choice == 'sclf4':
            model = sclf4
        else:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Predict using the selected model
        prediction = model.predict(input_df)
        print(f"Prediction: {prediction}")  # Log the prediction

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Log the error message
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
