import streamlit as st
from fastai.text.all import *

import pathlib
pathlib.PosixPath = pathlib.WindowsPath #linux->windows path 

st.set_page_config(page_title="Fake News Classifier")

st.title("Fake vs Real News Classifier")
st.write("Paste a news article below to classify it.")

@st.cache_resource # load model once
def load_model():
    return load_learner("models/fake_news_model.pkl")

learn = load_model() # load model into memory for usage

text = st.text_area("News article text", height=250)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        pred, idx, probs = learn.predict(text) # return predicted class, index of the class, probability per class
        st.success(f"Prediction: **{pred}**")
        st.write(f"Confidence: **{probs[idx]:.4f}**")
