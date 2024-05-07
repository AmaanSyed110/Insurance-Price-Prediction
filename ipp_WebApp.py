# -*- coding: utf-8 -*-
"""
Created on Mon May  6 19:32:16 2024

@author: Amaan
"""

import numpy as np
import pickle
import streamlit as st 

loaded_model = pickle.load(open('D:/Git-Hub projects/Insurance-Price-Prediction/trained_model.pkl','rb'))

def insurance_price_prediction(input_data):
  #changing input data to numpy array
  input_data_as_numpy_array=np.asarray(input_data, dtype=np.float64)
  #reshaping the array
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
  prediction = loaded_model.predict(input_data_reshaped)
  return(prediction[0])


def main():
  st.title ('Insurance Price Prediction Web App')

  age=st.text_input('Age')
  sex=st.text_input('Sex (Male=1, Female=0)')
  bmi=st.text_input('Body Mass Index')
  children=st.text_input('Number of Children')
  smoker=st.text_input('Smoker (Yes=1, No=0)')
  

  price= ""
  #code for prediction

  if st.button("Predict Price"):
    price=insurance_price_prediction([age,sex,bmi,children,smoker])
  
  st.success(price)


if __name__=='__main__':
  main()