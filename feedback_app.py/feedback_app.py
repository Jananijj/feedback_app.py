import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import smtplib
from email.mime.text import MIMEText
import pandas as pd
from datetime import datetime

# 🛠 Page config
st.set_page_config(page_title="AI Feedback Response System", layout="centered")

# 🚀 Load RoBERTa sentiment model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# 💬 Analyze sentiment
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs[0][0].numpy())
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[scores.argmax()]

# 🤖 Generate auto-reply
def auto_reply(sentiment):
    if sentiment == "Positive":
        return "✅ Thank you! We're happy to hear that you're satisfied with our service."
    elif sentiment == "Negative":
        return "⚠️ We're sorry to hear that. Your feedback will help us improve."
    else:
        return "ℹ️ Thank you for your honest input. We'll use this to serve you better."

# 📧 Send email to Anuja
def send_email(subject, body):
    sender_email = "anuja9feb@gmail.com"
    receiver_email = "anuja9feb@gmail.com"
    password = "ebdo jhyk qjju fczx"  # 🔐 Replace this with your 16-digit Gmail App Password (no spaces)

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
            return True
    except Exception as e:
        print("Email failed:", e)
        return False

# 🧾 Save to CSV (optional log)
def log_feedback(feedback, sentiment, reply):
    data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Feedback": [feedback],
        "Sentiment": [sentiment],
        "Reply": [reply]
    }
    df = pd.DataFrame(data)
    df.to_csv("all_feedback.csv", mode='a', header=not pd.io.common.file_exists("all_feedback.csv"), index=False)

# 🌐 Streamlit App UI
st.title("💬 AI Feedback Response System for IT Services")



