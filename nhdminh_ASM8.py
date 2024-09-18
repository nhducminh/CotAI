# REPOSITORY LINK:

# APP PUBLIC LINK:

import pickle
import numpy as np
import streamlit  as st
from sklearn.linear_model import LinearRegression

with open('model.pickle', 'rb') as f:
  model = pickle.load(f)

inputRadio = 0
inputTV = 0

col1, col2 = st.columns(2)
col2.header('Radio')
with col1:
  inputRadio = st.number_input("Insert a number")

col2.header('TV')
with col2:
  inputTV = st.number_input("Insert a number")

# if st.button("Predict", type="primary"):
#   if inputRadio*inputTV!=0:
#     y_input = np.array([[inputTV,inputRadio]])
#     y_input_predict = model.predict(y_input)
#     txt = st.text_area(f'Sale prediction: {y_input_predict}')
#   else:
#     txt = st.text_area(f'Please input TV and Radio buget')
