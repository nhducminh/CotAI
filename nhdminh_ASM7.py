
import requests
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import streamlit as st


def loadImg(URL):
    from PIL import Image
    img_url = URL
    img_pil = Image.open(requests.get(img_url, stream=True).raw)
    # img_pil # ảnh lưu ở format Pillow
    return np.array(img_pil) # ảnh lưu ở dạng numpy array
    
def kMeansImg(URL,k):
    img = loadImg(URL)
    X = img.reshape(img.shape[0]*img.shape[1],img.shape[2])
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X)
    img_new= ([kmeans.cluster_centers_[kmeans.labels_[i]] for i in range(len(X))])
    img_new= np.array(img_new).reshape(img.shape)
    img_new = img_new.astype(np.uint8)    
    return Image.fromarray(img_new)

# img_url = "https://www.popsci.com/uploads/2023/05/15/ButterflyFamilyTree.png"
k = st.slider("Select a range of k", 3, 16)
img_url = st.text_input("Image URL", "https://www.popsci.com/uploads/2023/05/15/ButterflyFamilyTree.png")
if st.button('Say Hello'):
    img_show = kMeansImg(img_url,k)
    img_show

