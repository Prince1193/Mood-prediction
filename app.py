
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
import numpy as np
import threading
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Global variables to store models, label encoder, and dataset
user_models = {}
label_encoder = None
top_models = []
df_ag = None  # DataFrame for aggregated data

# Define the path for the models directory and dataset
models_dir = 'C:\\Users\\lenovo\\OneDrive\\Desktop\\Final project\\models'
dataset_path = 'C:\\Users\\lenovo\\OneDrive\\Desktop\\Final project\\aggregated_data\\aggregated_data.csv'

# Flag to ensure the dataset and models are loaded only once
data_loaded = False

# Load dataset function
def load_dataset():
    global df_ag
    if os.path.exists(dataset_path):
        df_ag = pd.read_csv(dataset_path)
    else:
        raise FileNotFoundError("Dataset file not found!")

# Load models function
def load_models():
    global top_models
    top_models_files = sorted(os.listdir(models_dir))[:50]  # Adjust as needed
    top_models = [joblib.load(os.path.join(models_dir, f)) for f in top_models_files]

# Load user models and label encoder from file
def load_user_models():
    global user_models, label_encoder
    label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
    else:
        raise FileNotFoundError("Label encoder file not found!")

    for filename in os.listdir(models_dir):
        if filename.endswith('_model.pkl'):
            user_id = int(filename.split('_')[0])
            user_models[user_id] = joblib.load(os.path.join(models_dir, filename))

# Background function to load data and models
def load_data_in_background():
    global data_loaded
    if not data_loaded:
        print("Loading data in the background...")
        load_dataset()
        load_user_models()
        load_models()
        data_loaded = True
        print("Data loaded!")

# Start loading the data when the app starts
def start_background_loading():
    threading.Thread(target=load_data_in_background).start()

# Function to preprocess data
def preprocess_data(data):
    global df_ag
    # Convert 'date' columns to datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    df_ag['date'] = pd.to_datetime(df_ag['date'], errors='coerce')

    data['active_hrs'] = data['active_minutes'] / 60
    data.drop(columns=['active_minutes'], inplace=True)
    
    # Map categorical variables
    data['location'] = data['location'].map({'Park': 0, 'Office': 1, 'Home': 2, 'Gym': 3, 'Other': 4})
    data['weather_conditions'] = data['weather_conditions'].map({'Clear': 0, 'Fog': 1, 'Snow': 2, 'Rain': 3})
    data['workout_type'] = data['workout_type'].map({'Walking': 0, 'Cycling': 1, 'Yoga': 2, 'Swimming': 3, 'Gym Workout': 4, 'Running': 5})
    
    # Handle the user ID
    user_id = data['user_id'].iloc[0]
    
    if user_id in df_ag['user_id'].unique():
        # Get the most recent date for the user
        last_entry_date = df_ag[df_ag['user_id'] == user_id]['date'].max()
        # Compute the difference in days
        if pd.notna(last_entry_date):
            data['days_since_last_workout'] = (data['date'] - last_entry_date).dt.days
        else:
            data['days_since_last_workout'] = 0
    else:
        data['days_since_last_workout'] = 0

    return data

# Predict mood for existing user
def predict_existing_user_mood(user_id, data):
    model = user_models.get(user_id)
    if model:
        X = preprocess_data(data)
        X = X.drop(columns=['user_id', 'date'])
        required_columns = ['steps', 'calories_burned', 'distance_km', 'active_hrs',
                    'sleep_hours', 'heart_rate_avg', 'workout_type','weather_conditions', 'location', 'days_since_last_workout']
        X = X[required_columns]
        if not X.empty:
            prediction = model.predict(X)
            mood_prediction = label_encoder.inverse_transform([prediction[0]])[0]
            return int(mood_prediction) if isinstance(mood_prediction, np.int32) else mood_prediction
    return 'Model not found or no data'

# Predict mood for new user using top models
def predict_new_user_mood(data):
    X = preprocess_data(data)
    X = X.drop(columns=['user_id', 'date'])
    predictions = []
    for model in top_models:
        prediction = model.predict(X)
        mood_prediction = label_encoder.inverse_transform([prediction[0]])[0]
        predictions.append(int(mood_prediction) if isinstance(mood_prediction, np.int32) else mood_prediction)
    
    if predictions:
        # Use majority voting
        return max(set(predictions), key=predictions.count)
    return 'No predictions'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        user_id = int(form_data['user_id'])
        date = form_data['date']
        steps = float(form_data['steps'])
        calories_burned = float(form_data['calories_burned'])
        distance_km = float(form_data['distance_km'])
        active_minutes = int(form_data['active_minutes'])
        sleep_hours = float(form_data['sleep_hours'])
        heart_rate_avg = int(form_data['heart_rate_avg'])
        workout_type = form_data['workout_type']
        weather_conditions = form_data['weather_conditions']
        location = form_data['location']

        new_data = pd.DataFrame({
            'user_id': [user_id],
            'date': [date],
            'steps': [steps],
            'calories_burned': [calories_burned],
            'distance_km': [distance_km],
            'active_minutes': [active_minutes],
            'sleep_hours': [sleep_hours],
            'heart_rate_avg': [heart_rate_avg],
            'workout_type': [workout_type],
            'weather_conditions': [weather_conditions],
            'location': [location]
        })

        # Check if user_id is in existing user models
        if user_id in user_models:
            mood_prediction = predict_existing_user_mood(user_id, new_data)
        else:
            mood_prediction = predict_new_user_mood(new_data)
        
        return jsonify({'mood_prediction': mood_prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    start_background_loading()  # Start loading the data when the app starts
    app.run(debug=True)
