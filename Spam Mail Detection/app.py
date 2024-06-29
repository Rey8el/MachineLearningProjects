import pickle 
import streamlit as st 
with open("SpamDetectionModel.pkl", "rb") as file:
    classifier = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    cv = pickle.load(file)

def main():
    st.title("Spam Email Detection")
    text = st.text_area("Enter the text here: ")
    vectorized = cv.transform([text])
    result = classifier.predict(vectorized)
    if st.button("Detect"):
        if result == 1:
            st.error("SPAM")
        else:
            st.error("NOT SPAM")

main()
