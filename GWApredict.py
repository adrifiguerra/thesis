import streamlit as st
import numpy as np
import pickle

# Load the dataset for Random Forrest and Naive Bayes
loaded_model_nb = pickle.load(open('C:/Users/User/Desktop/Mayk/trained_model_nb.sav', 'rb'))
loaded_model_rf = pickle.load(open('C:/Users/User/Desktop/Mayk/trained_model_rf.sav', 'rb'))

# Creating a function for prediction

def GWA_prediction_nb(input_data):

    #Changing the input_data as numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_as_numpy_array = input_data_as_numpy_array.astype(np.float64)

    #reshape array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    # Predict the label using Naive Bayes
    prediction_nb = loaded_model_nb.predict(input_data_reshaped)

    # Print the predictions
    # For Naive Bayes
    if (prediction_nb == 1):
        return "Using Naive Bayes Algorithm, the student will have a GWA between 1.0000 - 1.2000."
    elif (prediction_nb == 2):
        return "Using Naive Bayes Algorithm, the student will have a GWA between 1.2001 - 1.4500."
    elif (prediction_nb == 3):
        return "Using Naive Bayes Algorithm, the student will have a GWA between 1.4501 - 1.7500."
    elif (prediction_nb == 4):
        return "Using Naive Bayes Algorithm, the student will have a GWA between 1.7501 - 2.0000."
    else:
        return "Using Naive Bayes Algorithm, the student will have a GWA between 2.0000 - 3.0000."
    

def GWA_prediction_rf(input_data):

    #Changing the input_data as numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_as_numpy_array = input_data_as_numpy_array.astype(np.float64)

    #reshape array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # Predict the label using Random Forest
    prediction_rf = loaded_model_rf.predict(input_data_reshaped)

    # For Random Forrest
    if (prediction_rf == 1):
        return "Using Random Forrest Algorithm, the student will have a GWA between 1.0000 - 1.2000."
    elif (prediction_rf == 2):
        return "Using Random Forrest Algorithm, the student will have a GWA between 1.2001 - 1.4500."
    elif (prediction_rf == 3):
        return "Using Random Forrest Algorithm, the student will have a GWA between 1.4501 - 1.7500."
    elif (prediction_rf == 4):
        return "Using Random Forrest Algorithm, the student will have a GWA between 1.7501 - 2.0000."
    else:
        return "Using Random Forrest Algorithm, the student will have a GWA between 2.0000 - 3.0000."

def main():

    # Giving Title
    st.title('General Weighted Average (GWA) Prediction Web Application')

    # Getting the input data from the user
    Gender = float(st.text_input("Gender (0 if Male, 1 if Female):"))
    School = float(st.text_input("Type of school (0 if Public, 1 if Private):"))
    Track = float(st.text_input("School Track (0 if Non-STEM, 1 if STEM):"))
    PUPCET = float(st.text_input("PUPCET Score:"))
    SHSGPA = float(st.text_input("SHS GPA (Grade Point Average):"))

    # Code for Prediction
    Predictnb = ''
    Predictrf = ''

    # Creating a button for Prediction
    if st.button('GWA reveal'):
        Predictnb = GWA_prediction_nb([Gender, School, Track, PUPCET, SHSGPA])
        Predictrf = GWA_prediction_rf([Gender, School, Track, PUPCET, SHSGPA])

    st.success(Predictnb)
    st.success(Predictrf)

if __name__ == '__main__':
    main()