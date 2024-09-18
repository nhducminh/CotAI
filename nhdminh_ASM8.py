import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

###########################

col1, col2 = st.columns([1,3])
with col1:
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        with col2:
            tab1, tab2 = st.tabs(["Train", "Inference"])
            with tab1:
                col3,col4 = st.columns(2)
                with col3:
                    st.dataframe(df)
                with col4:
                    options = st.multiselect(
                        "Choose feature",
                        (df.columns[0], df.columns[1], df.columns[2]),
                        max_selections=2)
                    
                    # YOUR CODE HERE
                    y = df.Sales
                    model = LinearRegression()
                    try:
                        if len(options)==1:                        
                            X = np.array(df.loc[:,options[0]])
                            X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=30)
                            model.fit(X_train,y_train)
                            y_predict = model.predict(X_test)
                            model_mae = mae(y_test,y_predict)
                            model_mse = mse(y_test,y_predict)
                            st.write(model_mae,model_mse)
                        elif len(options)==2:
                            X = np.array(df.loc[:,[options[0],options[1]]])
                            X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=30)
                            model.fit(X_train,y_train)
                            y_predict = model.predict(X_test)
                            model_mae = mae(y_test,y_predict)
                            model_mse = mse(y_test,y_predict)
                            st.write(model_mae,model_mse)
                            
                    except Exception as e:
                        st.write(e)

                    pass
                pass
            with tab2:
                pass


