import streamlit as st
import pickle

# Load vectorizer and model
with open('tfidf.pkl', 'rb') as f:
    Tfi = pickle.load(f)

with open('text_MLP.pkl', 'rb') as f:
    model = pickle.load(f)

def clean_text(text):
    return text.lower().strip()

def predict_intent(prompt):
    cleaned_prompt = clean_text(prompt)
    vectorized_prompt = Tfi.transform([cleaned_prompt])
    predicted_intent = model.predict(vectorized_prompt)[0]
    return predicted_intent

# Streamlit UI
st.set_page_config(page_title="Intent Classifier", page_icon="🧠")
st.title("🧠 Text Intent Classifier")
st.markdown("Enter a prompt below to classify its intent.")

custom_prompt = st.text_input("Enter your prompt:")

if custom_prompt:
    predicted_label = predict_intent(custom_prompt)
    st.success(f"**Predicted Intent:** `{predicted_label}`")
