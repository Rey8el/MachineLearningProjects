import pickle 
import pandas as pd
import numpy as np
import streamlit as st 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained classifier and vectorizer
with open("PhishingDomainDetection.pkl", "rb") as file:
    classifier = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    cv = pickle.load(file)

# Define tokenizer and stemmer
tokenizer = RegexpTokenizer(r"[A-Za-z]+")
stemmer = SnowballStemmer("english")

def main():
    st.title("Phishing Domain Detection")
    text = st.text_area("Enter the text here: ")
    
    tokens = tokenizer.tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    
    vectorized_text = cv.transform([stemmed_text])

    result = classifier.predict(vectorized_text)
    if st.button("Detect"):
        if result[0] == 0:
            st.error("SAFE")
        else:
            st.error("NOT SAFE")

main()
