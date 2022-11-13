# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
from pickle import dump
from pickle import load
import pickle


data = pd.read_csv('pronostico_dataset.csv')
array = data.values
X = array[:,0:-1]

loaded_model = load(open('SVC.sav','rb'))


# creating function for prediction
def predict(input_data):
    
    # changing the input data to numpy array 
    input_data_as_numpy_array = np.asarray(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 1):
        return 'Retinopathy Found'
    else:
        return 'Retinopathy Not Found'
    

def main():
        
        # giving a title
        st.title('Retinopathy Diabetic Predictor')
        
        # getting the input data from the user
        
        number_1 = st.number_input('Insert the AGE of the Patient')
        number_2 = st.number_input('Insert the Systolic_Bp of the Patient')
        number_3 = st.number_input('Insert the Diastolic_Bp of the Patient')
        number_4 = st.number_input('Insert the Cholesterol of the Patient')
        
        
        # code for Prediction
        diagnosis = ''
        
        # creating a button for Prediction
        if st.button('Diabetes Test Result'):
            diagnosis = predict([number_1,number_2,number_3,number_4])
            
        st.success(diagnosis)
        
        
        
if __name__ == '__main__':
    main()