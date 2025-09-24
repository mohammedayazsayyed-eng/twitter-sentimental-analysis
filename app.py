import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import snscrape.modules.twitter as sntwitter  # ✅ Replaces Nitter

# -------------------------------
# Download stopwords once
# -------------------------------
nltk.download('stopwords')
stop_words = stopwords.words('english')
port_stem = PorterStemmer()

# -------------------------------
# Load model and vectorizer
# -------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    with open('trained_model.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# -------------------------------
# Preprocessing function (same as Colab)
# -------------------------------
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

# -------------------------------
# Predict sentiment
# -------------------------------
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    transformed_text = vectorizer.transform([processed_text])
    prediction = model.predict(transformed_text)
    return "Negative" if prediction[0] == 0 else "Positive"

# -------------------------------
# Fetch Tweets using snscrape
# -------------------------------
def fetch_tweets(username, limit=5):
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterUserScraper(username).get_items()):
            if i >= limit:
                break
            tweets.append(tweet.content)
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
    return tweets

# -------------------------------
# Helper: Create HTML Card
# -------------------------------
def create_card(tweet_text, sentiment):
    color = "#28a745" if sentiment == "Positive" else "#dc3545"  # Green for positive, red for negative
    card_html = f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h5 style="color: white; margin: 0;">{sentiment} Sentiment</h5>
        <p style="color: white; margin-top: 5px;">{tweet_text}</p>
    </div>
    """
    return card_html

# -------------------------------
# Main App
# -------------------------------
def main():
    st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")
    st.title("🐦 Twitter Sentiment Analysis")

    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            if text_input.strip():
                sentiment = predict_sentiment(text_input)
                st.success(f"Sentiment: {sentiment}") if sentiment == "Positive" else st.error(f"Sentiment: {sentiment}")
            else:
                st.warning("Please enter some text first.")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username")
        if st.button("Fetch Tweets"):
            tweets_data = fetch_tweets(username)
            if tweets_data:
                for tweet_text in tweets_data:
                    sentiment = predict_sentiment(tweet_text)
                    st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
            else:
                st.warning("No tweets found for this user.")

if __name__ == "__main__":
    main()
