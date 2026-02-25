import streamlit as st
import numpy as np
import pickle
import json
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.corpus import stopwords
import nltk

# ==============================
# CONFIG
# ==============================
MAX_LEN = 200
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

st.title("Toxic Comment Classifier")
st.markdown("Detect toxic, threatening, obscene, insulting, and hateful comments")

# ==============================
# NLTK stopwords
# ==============================
@st.cache_resource
def download_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

STOP_WORDS = download_stopwords()

# ==============================
# Load tokenizer and thresholds
# ==============================
@st.cache_resource
def load_tokenizer():
    with open('processed/tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_thresholds():
    with open('models/thresholds.json', 'r') as f:
        return json.load(f)

tokenizer = load_tokenizer()
thresholds = load_thresholds()

# ==============================
# Load model
# ==============================
@st.cache_resource
def load_bilstm_model():
    return load_model("models/bilstm.keras", compile=False)

model = load_bilstm_model()

# ==============================
# Text cleaning
# ==============================
def clean_text(text):
    if not text:
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS]
    return ' '.join(words)

# ==============================
# Prediction
# ==============================
def predict_toxicity(text):
    cleaned = clean_text(text)
    if not cleaned:
        return None
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    preds = model.predict(padded, verbose=0)[0]
    results = {}
    for i, label in enumerate(LABELS):
        prob = float(preds[i])
        thresh = thresholds.get(label, 0.5)
        results[label] = {'probability': prob, 'prediction': prob > thresh}
    return results

# ==============================
# User input
# ==============================
user_input = st.text_area("Enter comment:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a comment")
    else:
        with st.spinner("Analyzing..."):
            results = predict_toxicity(user_input)
        if results:
            is_toxic = any(r['prediction'] for r in results.values())
            if is_toxic:
                st.error("⚠️ Toxic content detected!")
            else:
                st.success("✅ Comment appears safe")
            st.markdown("### Detailed Results")
            for label, r in results.items():
                st.write(f"{label.replace('_',' ').title()}: {r['probability']*100:.2f}% {'⚠️' if r['prediction'] else '✅'}")
        else:
            st.info("Text could not be processed.")