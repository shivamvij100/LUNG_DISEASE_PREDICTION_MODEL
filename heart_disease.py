import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import time

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
st.image('https://img.freepik.com/premium-vector/lungs-human-anatomy-icon-isolated-white-background_53562-14619.jpg')

# Load trained model
with open('lung_disease_pred.pkl','rb') as f:
    model = pickle.load(f)

# Load lung disease dataset (replace with your dataset URL)
url = '''https://github.com/ankitmisk/Lung_Disease_Prediction_ML_Model/blob/main/lung.csv?raw=true'''
df = pd.read_csv(url)

# Sidebar for user input
st.sidebar.header('Select Features to Predict Lung Disease')
st.sidebar.image('https://cdn-icons-png.flaticon.com/512/3022/3022248.png', width=150)

all_values = []

for i in df.iloc[:,:-1]:  # Loop through all feature columns except target
    min_value, max_value = df[i].agg(['min','max'])
    var = st.sidebar.slider(
        f'Select {i} value', 
        int(min_value), 
        int(max_value), 
        random.randint(int(min_value), int(max_value))
    )
    all_values.append(var)

final_value = [all_values]

# Make prediction
ans = model.predict(final_value)[0]

# Progress bar animation
random.seed(132)
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Lung Disease') 

place = st.empty()
place.image('https://i.makeagif.com/media/1-17-2024/dw-jXM.gif', width=200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

# Display results
if ans == 0:
    body = 'No Lung Disease Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = 'Lung Disease Found'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)

st.markdown('Designed by: **SHIVAM VIJ**')
