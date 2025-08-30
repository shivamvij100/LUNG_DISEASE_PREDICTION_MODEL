# The rewritten Streamlit app code.
# Save this as app.py and run with `streamlit run app.py`
# Ensure 'lung.csv' and 'lung_disease_pred.pkl' are in the same directory.

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import time
import io  # Not needed if using file

st.header('Lung Disease Prediction Using Machine Learning')

data = '''Lung Disease Prediction using Machine Learning

Lung disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes. 

In this project, I analyzed a lung disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of lung disease.

Algorithms Used:

**Logistic Regression**

**Naive Bayes**

**Support Vector Machine (Linear)**

**K-Nearest Neighbors**

**Decision Tree**

**Random Forest**

**XGBoost**

**Artificial Neural Network (1 Hidden Layer, Keras)**
'''

st.markdown(data)

# Update image to a lung-related illustration
st.image('https://media.istockphoto.com/id/1192036561/vector/cute-cartoon-funny-lungs-character-with-dumbbells-strong-smiling-lung-human-respiratory.jpg?s=612x612&w=0&k=20&c=O5FXi6cGHSkwmGv1xtJNBQXLI3BUerRHTnD9Kg2Ndn8=')

# Load trained model
with open('lung_disease_pred.pkl','rb') as f:
    model = pickle.load(f)

# Load lung disease dataset from local file
df = pd.read_csv('lung.csv')

# Sidebar for user input
st.sidebar.header('Select Features to Predict Lung Disease')
st.sidebar.image('https://cdn-icons-png.flaticon.com/512/3022/3022248.png', width=150)

all_values = []

for col in df.columns[:-1]:  # Loop through all feature columns except target
    min_value = df[col].min()
    max_value = df[col].max()
    is_float = df[col].dtype == 'float64'
    
    if is_float:
        step = 0.1
        default_value = round(random.uniform(min_value, max_value), 1)
    else:
        step = 1
        min_value = int(min_value)
        max_value = int(max_value)
        default_value = random.randint(min_value, max_value)
    
    var = st.sidebar.slider(
        f'Select {col} value', 
        min_value=min_value, 
        max_value=max_value, 
        value=default_value,
        step=step
    )
    all_values.append(var)

final_value = [all_values]

# Make prediction
# --- Make prediction with probabilities ---
# --- Make prediction with probabilities ---
prob = model.predict_proba(final_value)[0]  # [prob_no_disease, prob_disease]

# Let user adjust threshold in sidebar
threshold = st.sidebar.slider(
    "Select Threshold for Disease Prediction", 
    min_value=0.1, max_value=0.9, value=0.5, step=0.05
)

# Apply threshold
ans = 1 if prob[1] > threshold else 0

# Progress bar animation
random.seed(132)
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Lung Disease') 

place = st.empty()
place.image('https://www.shutterstock.com/image-vector/cartoon-sad-suffering-sick-lungs-600nw-2238775831.jpg', width=200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

# Display results
if ans ==
