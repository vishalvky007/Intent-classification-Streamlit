import streamlit as st
import pickle

# Load vectorizer and model
with open('tfidf.pkl', 'rb') as f:
    Tfi = pickle.load(f)

with open('text_MLP.pkl', 'rb') as f:
    model = pickle.load(f)

# Intent class labels
class_labels = [
    "find_object", "describe_environment", "read_document",
    "currency_recognition", "scan_qr", "colour_detection",
    "face_recognition", "no_class"
]

# Example prompts for each class
examples = {
    "find_object": "Where can I find my scooter keys?",
    "currency_recognition": "Can you tell if this is a new 2000 rupee note?",
    "read_document": "What's written on this electricity bill?",
    "face_recognition": "Who is this person in the Aadhaar photo?",
    "scan_qr": "Scan this Paytm QR code.",
    "colour_detection": "What color is this saree?",
    "describe_environment": "Describe what's around me in the market.",
    "no_class": "Play some music for me."
}

# Text preprocessing
def clean_text(text):
    return text.lower().strip()

# Intent prediction
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

# Submit button for prediction
if st.button("Submit"):
    if custom_prompt:
        predicted_label = predict_intent(custom_prompt)
        st.success(f"**Predicted Intent:** `{predicted_label}`")
    else:
        st.warning("⚠️ Please enter a prompt before submitting.")

# Display all possible class labels
with st.expander("🔖 View Possible Intent Labels"):
    st.markdown("### List of Possible Intents:")
    for label in class_labels:
        st.markdown(f"- `{label}`")

# Display example prompts for each class
with st.expander("🧪 View Example Test Prompts"):
    st.markdown("### Example Prompts for Each Intent:")
    for label, prompt in examples.items():
        st.markdown(f"**{label}**: _{prompt}_")
