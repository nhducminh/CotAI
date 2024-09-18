import streamlit as st
import pandas as pd
from io import StringIO

col1, col2 = st.columns([1,3])
with col1:
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)
        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        with col2:
            tab1, tab2 = st.tabs(["Train", "Inference"])
            with tab1:
                st.dataframe(df)
                pass
            with tab2:
                pass


