import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

###########################
st.set_page_config(layout="wide")

col1_main, col2_main = st.columns([1,3])
with col1_main:
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
        with col2_main:
            tab1, tab2 = st.tabs(["Train", "Inference"])
            with tab1:
                col1_tab1, col2_tab1 = st.columns([1,3])
                with col1_tab1:
                    st.dataframe(df)
                with col2_tab1:
                    try:
                        options = st.multiselect(
                            "Choose feature",
                            (df.columns[0], df.columns[1], df.columns[2]),
                            max_selections=3)
                        y = df.Sales
                        model = LinearRegression()
    
                        X = np.array(df.loc[:,[x for x in options]]).reshape(-1,len(options))
                        
                        X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=30)
                        
                        model.fit(X_train,y_train)             
                        w = model.coef_
                        b = model.intercept_
                        y_predict = model.predict(X_test)      
                        st.write(f"Model trained:")
                        st.write(f"MAE:{mae(y_test,y_predict)}")
                        st.write(f"MSE: {mse(y_test,y_predict)}")
                        if len(options)==1:                        
                            fig2d = go.Figure(data=[go.Scatter(x=X,y=y,mode='markers'),
                                                  go.Scatter(x=X, y=X*w+b,mode='lines')])
                            st.plotly_chart(fig2d)
    
                        if len(options)==2:
                            X_space = np.linspace(np.min(X[:,0]),np.max(X[:,0]))
                            Y_space = np.linspace(np.min(X[:,1]),np.max(X[:,1]))
                            
                            XX,YY = np.meshgrid(X_space,Y_space)
                            XY = np.c_[XX.ravel(),YY.ravel()]
                            Z = XY@w + b                            
                            fig3d = go.Figure(data=[ go.Surface(x = X_space,y = Y_space, z = Z.reshape(XX.shape)),
                                                  go.Scatter3d(x = X[:,0],y = X[:,1], z = y, mode = 'markers')])
                            st.plotly_chart(fig3d)
                        
                        with tab2:
                            input = [st.number_input(f"Insert {x} number") for x in options ]
                            if st.button("Predict", type="primary"):     
                                if np.all(np.array(input)):
                                    output_predict = model.predict(np.array(input).reshape(-1,len(options)))
                                    st.write(f"Prediction {output_predict}")
                                else:
                                    st.write("Please input")
                    except:
                        pass
             
                pass
            with tab2:
                pass


