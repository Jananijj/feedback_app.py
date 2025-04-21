import streamlit as st
st.set_page_config(page_title="AI Feedback Assistant")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import smtplib
from email.mime.text import MIMEText

# Load RoBERTa model from Hugging Face
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

# Auto-reply based on sentiment
def auto_reply(sentiment):
    if sentiment == "Positive":
        return "✅ Thank you! We're happy to hear that you're satisfied with our service."
    elif sentiment == "Negative":
        return "⚠️ We're sorry to hear that. Your feedback will help us improve."
    else:
        return "ℹ️ Thank you for your honest input. We'll use this to serve you better."

# Send email with feedback + AI reply
def send_email(subject, body):
    sender_email = "your_email@gmail.com"
    receiver_email = "your_email@gmail.com"
    password = "your_app_password_here"  # App Password (16-digit)

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
        print("Email error:", e)
        return False

# Streamlit UI
st.title("💬 AI Feedback Response System for IT Services")
st.markdown("This system accepts feedback from clients or users, analyzes it using AI, and generates a professional auto-reply. You will also receive an email notification of the feedback and response.")

feedback = st.text_area("📝 Enter feedback here:")

if st.button("Generate AI Reply & Send Email"):
    if feedback.strip():
        sentiment = analyze_sentiment(feedback)
        reply = auto_reply(sentiment)

        # Compose email body
        body = f"""
📩 New Feedback Received:

🗒 Feedback: {feedback}

🤖 Detected Sentiment: {sentiment}
📩 Auto-Reply: {reply}
        """

        # Send email
        sent = send_email("New AI Feedback Submission", body)

        # Display on screen
        st.success(f"🤖 Sentiment: {sentiment}")
        st.info(f"📩 Auto-Reply: {reply}")

        if sent:
            st.success("📬 Email sent successfully to your inbox!")
        else:
            st.error("❌ Failed to send email.")
    else:
        st.warning("Please enter feedback before submitting.")


