import streamlit as st
import pandas as pd
from io import StringIO
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.utils import set_random_seed
from keras.backend import clear_session
from keras.models import load_model
import pickle


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'][::-1]
X_train = X_train / 255
X_test = X_test / 255



#  Mã hoá One-hot Encoding trên 2 tập train, test
nclass = 10
y_train_ohe = to_categorical(y_train,nclass)
y_test_ohe = to_categorical(y_test,nclass)
#  Tạo Keras model
clear_session()
set_random_seed(99)

modelname = 'fashion_mnist.keras'
model = Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(Flatten())
model.add(Dense(nclass,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])




tab1, tab2 = st.tabs(["Train", "Inference"])
with tab1:
    col1_1, col1_2 = st.columns([1,1])
    with col1_1:        
        #Load Dataset and visualize
        st.write('Dataset fashion_mnist')
        fig, axs = plt.subplots(10, 10)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        for i in range(10):
            ids = np.where(y_train == i)[0]
            for j in range(10):
                target = np.random.choice(ids)
                axs[i][j].axis('off')
                axs[i][j].imshow(X_train[target], cmap='gray')
        st.pyplot(fig)

    with col1_2:
        #set epochs and train
        epochs = st.slider("Select epochs number", 5, 200)
        if st.button("Train", type="primary"):
            with st.spinner('Model training'):
                history = model.fit(X_train,y_train_ohe,epochs=epochs,verbose=0)
                model.save(modelname)
            loss, accuracy = model.evaluate(X_test, y_test_ohe)                
            st.write(f"Model trained and saved. Test accuracy {accuracy*100:.2f}%")

            plt.figure(figsize=(5,4))
            plt.ylabel('Loss/Accuracy')
            plt.xlabel('Epochs')
            plt.title('Learning Curve')

            plt.plot(history.history['accuracy'])
            plt.plot(history.history['loss'])
            plt.legend(['accuracy','loss'])
            st.pyplot(plt)


with tab2:
    col2_1, col2_2 = st.columns([3,1])
    uploaded_file = st.file_uploader("Choose a file",type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)                
        # To read file as bytes:
        with col2_1:                    
            st.image(image)    
        with col2_2:        
            try:                               
                image_np = image.resize((28,28))
                image_np = image_np.convert('L')
                image_np = np.array(image_np)/255
                image_np = image_np.reshape(1,28,28)
                model = load_model(modelname) 
                y_test_pred = model.predict(image_np)
                yLabel = np.argsort(y_test_pred)[0][-3:][::-1]


                labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'][::-1]
                st.write('Predict:')
                for s in yLabel:
                    st.write(f'{labels[s]}: {y_test_pred[0][s]*100:.2f},%')    
            except Exception as e:
                st.write('Model not found')                              
