import pickle
import numpy as np
import streamlit as st
from google.cloud import firestore
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore , auth
from streamlit_option_menu import option_menu

# Check if Firebase app is already initialized
if not firebase_admin._apps:
    # Initialize Firebase app
    cred = credentials.Certificate('my-app-6e7d6-firebase-adminsdk-xdfp1-28f6b1c202.json')
    firebase_admin.initialize_app(cred)

# Connect to Firestore database
db = firestore.client()
# Load the XGBoost model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Define function to make predictions
def predict(features):
    # Convert feature values to numeric format
    numeric_features = []
    for feature in features:
        try:
            numeric_features.append(float(feature))
        except ValueError:
            st.error(f"Invalid input: {feature}. Please enter a numeric value.")
            return None
    
    # Check if all features are converted successfully
    if len(numeric_features) != len(features):
        return None

    # Convert to numpy array and reshape
    input_features = np.array(numeric_features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)
    return int(prediction[0])  # Return prediction as integer (0 or 1)
    

from PIL import Image
# Loading Image using PIL
im = Image.open('new.png')
# Adding Image to web app
st.set_page_config(page_title="CardioCare", page_icon = im)

# Title of the web app
st.title('CardioCare - A Cardiovascular Disease Analysis')

# Sidebar options
with st.sidebar:
    selected = option_menu('CardioCare',
                           ['Risk Identification',
                            'Patient Records'],
                           icons=['heart', 'person'],
                           default_index=0)

# Display form for heart disease prediction
if selected == 'Risk Identification':
    st.header('Enter Information')
    # Layout adjustments: Name and Age on the same row
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input('Please enter your name:', key="name")
    with col2:
        age = st.text_input('Age', key="age")

    def validate_input(value, min_value, max_value):
        if value < min_value or value > max_value:
            st.error(f"Please enter a value between {min_value} and {max_value}.")
            return False
        return True

    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox(
            "Sex (0: Male, 1: Female)",
            options=[(0, '0'), (1, '1')],
            format_func=lambda x: x[1]  # Display the string part only, including the numerical code
        )
        sex = sex[0]  # Extract the numerical code directly for use in the model
    with col2:
        resting_systolic_bp = st.number_input('Resting Systolic Blood Pressure', min_value=80, max_value=180)
        if not validate_input(resting_systolic_bp, 80, 180):
            resting_systolic_bp = None

    col1, col2 = st.columns(2)
    with col1:
        resting_diastolic_bp = st.number_input('Resting Diastolic Blood Pressure', min_value=60, max_value=115)
        if not validate_input(resting_diastolic_bp, 60, 115):
            resting_diastolic_bp = None
    with col2:
        cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=120, max_value=340)
        if not validate_input(cholesterol, 120, 340):
            cholesterol = None

    col1, col2 = st.columns(2)

    # Add the line below
    with col1:
        # Add the line above
        oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=3.1, step=0.1, format="%.1f")
        if not validate_input(oldpeak, 0.0, 3.1):
            oldpeak = None

    with col2:
        thalach = st.number_input('Maximum Heart Rate', min_value=85, max_value=200)
        if not validate_input(thalach, 85, 200):
            thalach = None

    # Chest pain type input as dropdown
    chest_pain_input = st.selectbox('Chest Pain Type (0: asymptomatic, 1: atypical angina, 2: non-anginal pain, 3: typical angina)', ['0 ', '1', '2', '3'])

    # Resting ECG input as dropdown
    restecg = st.selectbox('Resting ECG (0: Normal, 1: ST-T wave abnormality, 2: LVH)', ['0', '1', '2'])

    # Thalassemia input as dropdown
    thalassemia = st.selectbox('Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversible Defect ', ['0', '1', '2'])

    # Slope input as dropdown
    slope = st.selectbox('Slope of the Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)', ['0', '1', '2'])

    # BMI input as dropdown
    bmi = st.selectbox('BMI (0: Underweight, 1: Normal weight, 2: Overweight, 3: Obese)', ['0', '1', '2', '3'])

    fasting_blood_sugar = st.selectbox('Fasting Blood Sugar (1: Yes, 0: No)', ['0', '1'])
    exang = st.selectbox('Exercise Induced Angina (1: Yes, 0: No)', ['0', '1'])
    stroke = st.selectbox('Stroke (1: Yes, 0: No)', ['0', '1'])
    smoking = st.selectbox('Smoking (1: Yes, 0: No)', ['0', '1'])
    family_history = st.selectbox('Family History (1: Yes, 0: No)', ['0', '1'])
    shortness_of_breath = st.selectbox('Shortness of Breath (1: Yes, 0: No)', ['0', '1'])
    palpitations = st.selectbox('Palpitations (1: Yes, 0: No)', ['0', '1'])
    ca = st.selectbox('Ca (No. of blocked vessels)', ['0', '1', '2', '3'])

    # Predict button
    if st.button('Predict'):
        # Collect input features
        features = [age, sex, chest_pain_input, resting_systolic_bp, resting_diastolic_bp,
                    fasting_blood_sugar, cholesterol, restecg, thalach, exang, oldpeak,
                    slope, smoking, bmi, family_history, shortness_of_breath, palpitations,
                    ca, thalassemia, stroke]
        # Make prediction
        prediction = predict(features)
        # Display prediction
        if prediction is not None:
            if prediction == 1:
                st.error('Based on the provided information, you may have a risk of heart disease.')
                prediction_label = 1  # Yes
            else:
                st.success('Based on the provided information, you may not have a risk of heart disease.')
                prediction_label = 0  # No

    # Add Record button
    if st.button('Add Record', key="add_record_button"):
        # Collect input features
        features = [age, sex, chest_pain_input, resting_systolic_bp, resting_diastolic_bp,
                    fasting_blood_sugar, cholesterol, restecg, thalach, exang, oldpeak,
                    slope, smoking, bmi, family_history, shortness_of_breath, palpitations,
                    ca, thalassemia, stroke]
        # Make prediction
        prediction = predict(features)
        # Display prediction
        if prediction is not None:
            # Add record to Firestore with prediction result
            db.collection('patient_data').add({
                'name': name,
                'age': age,
                'sex': sex,
                'resting_systolic_bp': resting_systolic_bp,
                'resting_diastolic_bp': resting_diastolic_bp,
                'cholesterol': cholesterol,
                'oldpeak': '{:.1f}'.format(oldpeak),
                'thalach': thalach,
                'chest_pain_input': chest_pain_input,
                'restecg': restecg,
                'thalassemia': thalassemia,
                'slope': slope,
                'bmi': bmi,
                'fasting_blood_sugar': fasting_blood_sugar,
                'exang': exang,
                'stroke': stroke,
                'smoking': smoking,
                'family_history': family_history,
                'shortness_of_breath': shortness_of_breath,
                'palpitations': palpitations,
                'ca': ca,
                'prediction_result': prediction  # Include prediction result
            })

            # Show success message
            st.success("Record added successfully!")

# Display patient records
elif selected == 'Patient Records':
    # Show Data button 
    if st.button("Show Data"):
        # Retrieve data from Firestore
        docs = db.collection('patient_data').get()
        # Convert data to a list of dictionaries
        data = [doc.to_dict() for doc in docs]

        # Reorder data to match the sequence of input fields
        ordered_data = []
        for entry in data:
            ordered_entry = {
                'Name': entry['name'],
                'Age': entry['age'],
                'Sex': entry['sex'],
                'Resting Systolic BP': entry['resting_systolic_bp'],
                'Resting Diastolic BP': entry['resting_diastolic_bp'],
                'Cholesterol (mg/dl)': entry['cholesterol'],
                'Oldpeak': entry['oldpeak'],
                'Maximum Heart Rate': entry['thalach'],
                'Chest Pain Type': entry['chest_pain_input'],
                'Resting ECG': entry['restecg'],
                'Thalassemia': entry['thalassemia'],
                'Slope of Peak Exercise ST Segment': entry['slope'],
                'BMI': entry['bmi'],
                'Fasting Blood Sugar': entry['fasting_blood_sugar'],
                'Exercise Induced Angina': entry['exang'],
                'Stroke': entry['stroke'],
                'Smoking': entry['smoking'],
                'Family History': entry['family_history'],
                'Shortness of Breath': entry['shortness_of_breath'],
                'Palpitations': entry['palpitations'],
                'Ca (No. of blocked vessels)': entry['ca'],
                'Prediction Result': entry.get('prediction_result', '')  # Handle KeyError
            }
            ordered_data.append(ordered_entry)

        # Display data in table format on a new page
        st.write("## Patient Data")
        st.table(ordered_data)

    # Function to search for a record by name
    def search_record_by_name(name):
        # Query Firestore for records with matching name
        records_ref = db.collection('patient_data').where('name', '==', name).stream()
        results = [record.to_dict() for record in records_ref]
        return results

    # Function to delete a record by its name
    def delete_record_by_name(name):
        # Query Firestore for records with matching name
        records_ref = db.collection('patient_data').where('name', '==', name).stream()
        # Check if any records with the given name exist
        record_exists = False
        for record in records_ref:
            db.collection('patient_data').document(record.id).delete()
            record_exists = True
        if record_exists:
            st.write("Record(s) deleted successfully!")
        else:
            st.write("No records found with the given name.")

    # Text input field to enter the name of the record to search
    name = st.text_input('Enter the name of the record:')

    # Button to trigger search action
    if st.button('Search'):
        if name:
            search_results = search_record_by_name(name)
            if search_results:
                st.write("Search Results:")
                df = pd.DataFrame(search_results)
                # Reorder the columns to match the form sequence
                df = df[['name', 'age', 'sex', 'resting_systolic_bp', 'resting_diastolic_bp', 
                         'cholesterol', 'oldpeak', 'thalach', 'chest_pain_input', 'restecg', 
                         'thalassemia', 'slope', 'bmi', 'fasting_blood_sugar', 'exang', 
                         'stroke', 'smoking', 'family_history', 'shortness_of_breath', 
                         'palpitations', 'ca', 'prediction_result']]
                st.table(df)
            else:
                st.write("No records found.")
    
    # Text input field to enter the name of the record to delete
    delete_name = st.text_input('Enter the name of the record to delete:')

    # Button to trigger record deletion
    if st.button('Delete'):
        if delete_name:
            delete_record_by_name(delete_name)
        else:
            st.write("Please enter a name to delete a record.")