import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os

# Unzip the uploaded dataset
with zipfile.ZipFile("medquad.csv.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

# Load the dataset
df = pd.read_csv("medquad.csv")
df = df.dropna(subset=['question', 'answer'])

# TF-IDF setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question'])

# Streamlit UI
st.title("Deployed Chatbot")
st.write("Ask a medical question:")

user_input = st.text_input("Your question here:")

if st.button("Ask"):
    if user_input:
        user_tfidf = vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)
        best_match_idx = similarity_scores.argmax()
        best_question = df.iloc[best_match_idx]['question']
        best_answer = df.iloc[best_match_idx]['answer']

        st.subheader("Chatbot Response:")
        st.write(best_answer)

        st.markdown("---")
        st.markdown("*Matched Question:* " + best_question)
    else:
        st.warning("Please enter a question to get a response.")
