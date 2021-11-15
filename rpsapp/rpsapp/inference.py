import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from lifetimes import ParetoNBDFitter
import xgboost
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

def get_customerscore (frequency, recency, Age, Monetary):
    """Write a Function to Predict the Class Name

    Args:
        :param Age:
        :param recency:
        :param frequency:

    Returns:
        [str]: Prediction

    """
    ROOT_DIR = os.path.dirname (os.path.abspath (__file__))
    path = os.path.join (ROOT_DIR + '/static/model/')
    path_lkp = os.path.join (ROOT_DIR + '/static/lookupfiles/')
    customerconversion_model_file = path + 'lifetimes_paretonbdfitter.pkl'
    customerpersona_model_file = path + 'Persona_XGBoost_pipeline.joblib'
    customerpersona_lkp_model_file = path_lkp + 'Customer_Persona_lookup.csv'
    pnbd_fitter: ParetoNBDFitter = ParetoNBDFitter ()
    pnbd_fitter.load_model (customerconversion_model_file)
    expected_number_of_purchase = pnbd_fitter.predict (30, frequency=frequency, recency=recency, T=Age)
    probability_if_alive = pnbd_fitter.conditional_probability_alive (frequency=frequency, recency=recency, T=Age)
    lkp_customer_persona = pd.read_csv(customerpersona_lkp_model_file)
    lkp_customer_persona.set_index('Customer_Persona_code', inplace=True)
    customer_persona_predict = joblib.load (customerpersona_model_file)
    cp_predict = pd.DataFrame ({"f0": recency, "f1": frequency, "f2": Monetary}, index=[0])
    customer_persona_code_predictions = customer_persona_predict.predict(cp_predict)
    print(customer_persona_code_predictions.flatten()[0])
    print(lkp_customer_persona)
    customer_persona_predictions = lkp_customer_persona.loc[customer_persona_code_predictions.flatten()[0],
                                                            'Customer_Persona']
    print(customer_persona_predictions)
    return round (expected_number_of_purchase, 3), round (probability_if_alive, 3), customer_persona_predictions
