import streamlit as st
st.set_page_config(page_title="AI Feedback Assistant")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load model from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Analyze sentiment
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs[0][0].numpy())
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[scores.argmax()]

# Auto-reply logic
def auto_reply(sentiment):
    if sentiment == "Positive":
        return "✅ Thank you! We're happy to hear that you're satisfied with our service."
    elif sentiment == "Negative":
        return "⚠️ We're sorry to hear that. Your feedback will help us improve."
    else:
        return "ℹ️ Thank you for your honest input. We'll use this to serve you better."

# Streamlit App UI
st.title("💬 AI Feedback Response System for IT Services")
st.markdown("please share your feedback.")

feedback = st.text_area("📝 Enter your feedback here:")

if st.button("Submit"):
    if feedback.strip():
        sentiment = analyze_sentiment(feedback)
        reply = auto_reply(sentiment)
        st.success(f"🤖 Sentiment: {sentiment}")
        st.info(f"📩 Auto-Reply: {reply}")
    else:
        st.warning("Please enter feedback text.")

