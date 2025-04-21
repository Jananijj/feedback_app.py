import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import smtplib
from email.mime.text import MIMEText
import pandas as pd
from datetime import datetime

# Page setup
st.set_page_config(page_title="AI Feedback Response System", layout="centered")

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Analyze sentiment
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs[0][0].numpy())
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[scores.argmax()]

# Generate reply
def auto_reply(sentiment):
    if sentiment == "Positive":
        return "✅ Thank you! We're happy to hear that you're satisfied with our service."
    elif sentiment == "Negative":
        return "⚠️ We're sorry to hear that. Your feedback will help us improve."
    else:
        return "ℹ️ Thank you for your honest input. We'll use this to serve you better."

# Send email
def send_email(subject, body):
    sender_email = "anuja9feb@gmail.com"
    receiver_email = "anuja9feb@gmail.com"
    password = "your_16_digit_app_password_here"  # <-- paste your real app password here

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
            return True
    except Exception as e:
        print("Email failed:", e)
        return False

# Log to CSV
def log_feedback(name, feedback, sentiment, reply):
    data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Name": [name],
        "Feedback": [feedback],
        "Sentiment": [sentiment],
        "Reply": [reply]
    }
    df = pd.DataFrame(data)
    df.to_csv("all_feedback.csv", mode="a", header=not pd.io.common.file_exists("all_feedback.csv"), index=False)

# UI
st.title("💬 AI Feedback Response System for IT Services")
st.markdown("This AI tool analyzes feedback, detects sentiment, generates a reply, and emails the result to the admin.")

user_name = st.text_input("👤 Your Name")
feedback = st.text_area("✍️ Enter your feedback here")

if st.button("Generate AI Reply and Send Email"):
    if feedback.strip() and user_name.strip():
        sentiment = analyze_sentiment(feedback)
        reply = auto_reply(sentiment)

        st.success(f"🤖 Sentiment: {sentiment}")
        st.info(f"📩 Auto-Reply: {reply}")

        # Prepare email body
        email_body = f"""📥 New Feedback Received:

From: {user_name}
Feedback: {feedback}
Sentiment: {sentiment}
AI Reply: {reply}
        """

        # Send email
        sent = send_email("New Feedback Submission", email_body)
        log_feedback(user_name, feedback, sentiment, reply)

        if sent:
            st.success("✅ Email sent successfully to Anuja's inbox!")
        else:
            st.error("❌ Failed to send email. Please double-check your Gmail app password.")
    else:
        st.warning("Please enter both your name and feedback.")
