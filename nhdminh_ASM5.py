import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def f(x,mean,sd):
    return (1/(sd*(2*np.pi)**.5))*np.exp(-(x-mean)**2/2*sd**2)



values1 = st.slider("Select a range of Mean", -10., 10.0)
values2 = st.slider("Select a range of SD", 0.1, 5.0)

st.write("Mean:", values1)
st.write("Standard deviation:", values2)

# YOUR CODE HERE

mean = values1
sd = values2
x = np.linspace(-50,50)


fig, ax = plt.subplots()
ax.plot(x,f(x,mean,sd),label = 'Normal Distribution')
ax.title.set_text('Gauss Mass Function')
ax.set_xlabel("x")
ax.set_ylabel("p(x)")

ax.legend()



st.pyplot(fig)