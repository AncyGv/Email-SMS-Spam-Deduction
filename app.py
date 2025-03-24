import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Function to process text
def transform_text(text_message):
    text_message = text_message.lower()
    text_message = nltk.word_tokenize(text_message)

    y = [i for i in text_message if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI design
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")
st.markdown("""
    <style>
        body, .stApp { background-color: #a59ba8; color: #333333; }
        .main { background-color: #745a7d; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); }
        h1 { color: #7f5e8a; text-align: center; }
        .stTextArea textarea { background-color: #745a7d; color: black; border-radius: 8px; border: 1px solid #ccc; }
        .stButton button { background-color: #5b4363; color: white; border-radius: 8px; padding: 10px; font-weight: bold; }
        .stAlert { text-align: center; font-size: 18px; }
    </style>
    <div class='main'>
""", unsafe_allow_html=True)

# App Title
st.title("📧 Email/SMS Spam Classifier 🚨")
st.markdown("### Detect spam messages instantly with AI-powered analysis.")

# User Input
input_sms = st.text_area("✉ Enter your message here:")

# Predict Button
if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠ Please enter a message to analyze!")
    else:
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0]  # Get probability score

        # Visual Output
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3, 2))
        labels = ['Not Spam', 'Spam']
        colors = ['green', 'red']
        ax.bar(labels, prob, color=colors, alpha=0.7)
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        ax.set_title("Spam Probability Score")

        # Show Result
        if result == 1:
            st.error("🚫 This message is classified as SPAM!", icon="🚨")
            st.pyplot(fig)
        else:
            st.success("✅ This message is NOT Spam!", icon="✅")
            st.pyplot(fig)

# Footer
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("🔹 Developed with ❤ using Streamlit & NLTK 🔹")