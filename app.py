from twitter_fetch import get_tweets
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from utils import clean_text

from transformers import pipeline

# Load ML model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load BERT model
bert_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"   # 🔥 THIS FIXES YOUR ERROR
)

# Page config
st.set_page_config(page_title="Advanced Sentiment Analyzer", layout="centered")

# ---------- CUSTOM UI ----------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 40px;
    color: #00FFAA;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #AAAAAA;
}
.result {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 25px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">💬 Sentiment Analysis App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">👉 Please enter a one-line tweet below</div>', unsafe_allow_html=True)

# Input
user_input = st.text_area("Enter your text:", height=120)

# ---------- MODEL SELECTION ----------
model_choice = st.radio("Choose Model:", ["ML Model", "BERT Model"])

# ---------- ANALYZE ----------
if st.button("Analyze"):
    if user_input.strip() != "":
        
        cleaned = clean_text(user_input)

        # ML MODEL
        if model_choice == "ML Model":
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]

            result = "Positive 😊" if pred == 1 else "Negative 😠"

        # BERT MODEL
        else:
            bert_result = bert_model(user_input)[0]
            label = bert_result['label']

            if "POSITIVE" in label:
                result = "Positive 😊"
            else:
                result = "Negative 😠"

        # Display result
        color = "#1ABC9C" if "Positive" in result else "#E74C3C"
        st.markdown(f'<div class="result" style="background-color:{color};">{result}</div>', unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter a one-line tweet!")

# ---------- SAMPLE TEST ----------
st.subheader("🔹 Test Sample Tweets")

samples = [
    "I love this product!",
    "Worst experience ever",
    "This is amazing",
    "I hate this service"
]

for text in samples:
    if st.button(text):
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        result = "Positive 😊" if pred == 1 else "Negative 😠"
        st.write(f"{text} → {result}")

# ---------- DASHBOARD ----------
st.subheader("📊 Sentiment Dashboard")

data = pd.DataFrame({
    "Sentiment": ["Positive", "Negative"],
    "Count": [70, 30]  # Example values
})

fig = px.pie(data, names='Sentiment', values='Count', title="Sentiment Distribution")
st.plotly_chart(fig)

st.subheader("🐦 Live Twitter Sentiment")

query = st.text_input("Enter topic (e.g., AI, IPL, Movies):")

if st.button("Fetch Tweets"):
    tweets = get_tweets(query)

    if len(tweets) == 0:
        st.warning("No tweets found or API limit reached")
    else:
        for tweet in tweets:
            cleaned = clean_text(tweet)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]

            sentiment = "Positive 😊" if pred == 1 else "Negative 😠"

            st.write(f"👉 {tweet}")
            st.write(f"Sentiment: {sentiment}")
            st.write("---")