import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Create a sample dataset for training
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
data = {
    'experience': np.random.uniform(0, 30, n_samples),
    'salary': np.random.uniform(30000, 200000, n_samples),
    'projects': np.random.randint(1, 20, n_samples),
    'hours': np.random.uniform(35, 60, n_samples),
    'satisfaction': np.random.uniform(1, 10, n_samples)
}

# Create target variable (performance score)
performance = (
    0.3 * data['experience'] +
    0.2 * (data['salary'] / 100000) +
    0.25 * (data['projects'] / 10) +
    0.15 * (data['hours'] / 40) +
    0.1 * data['satisfaction']
) + np.random.normal(0, 0.5, n_samples)

# Normalize performance to 0-100 scale
performance = ((performance - performance.min()) / (performance.max() - performance.min())) * 100

# Create DataFrame
df = pd.DataFrame(data)
df['performance'] = performance

# Train XGBoost model
model = XGBRegressor(random_state=42)
X = df.drop('performance', axis=1)
y = df['performance']
model.fit(X, y)

def predict_performance(experience, salary, projects, hours, satisfaction):
    # Create input data
    input_data = pd.DataFrame({
        'experience': [experience],
        'salary': [salary],
        'projects': [projects],
        'hours': [hours],
        'satisfaction': [satisfaction]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return round(prediction, 2)

# Streamlit app
st.title('Employee Performance Prediction')

# Input fields
experience = st.number_input('Years of Experience', min_value=0.0, max_value=50.0, value=5.0)
salary = st.number_input('Current Salary (USD)', min_value=20000, max_value=500000, value=50000)
projects = st.number_input('Number of Projects', min_value=0, max_value=50, value=5)
hours = st.number_input('Average Working Hours/Week', min_value=20, max_value=80, value=40)
satisfaction = st.number_input('Job Satisfaction (1-10)', min_value=1.0, max_value=10.0, value=7.0)

if st.button('Predict Performance'):
    performance_score = predict_performance(experience, salary, projects, hours, satisfaction)
    st.success(f'Predicted Performance Score: {performance_score}')
    
    # Add interpretation
    st.subheader('Performance Analysis:')
    if performance_score >= 80:
        st.write('🌟 Outstanding Performance! This employee is performing exceptionally well.')
    elif performance_score >= 60:
        st.write('✅ Good Performance! The employee is meeting expectations.')
    elif performance_score >= 40:
        st.write('⚠️ Average Performance. There might be room for improvement.')
    else:
        st.write('❗ Below Average Performance. Consider performance improvement plan.')
