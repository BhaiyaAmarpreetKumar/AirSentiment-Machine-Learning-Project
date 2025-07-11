# MACHINE LEARNING PROJECT IMPLEMENTATION
# AirSentiment Analysis (Airline Tweet Sentiment Predictor)

import streamlit as st
import pandas as pd
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


# Downloading NLTK data 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Loading model and vectorizer

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Preprocessing function

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Streamlit App UI

st.set_page_config(page_title="Airline Sentiment", layout="wide")

# Header
st.markdown("<h1 style='text-align: center;'>✈️ Airline Tweet Sentiment Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Check whether a tweet is Positive, Neutral, or Negative</h4>", unsafe_allow_html=True)
st.markdown("---")

# logo
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Airplane_silhouette.svg/600px-Airplane_silhouette.svg.png", width=120)

# Input section
st.subheader("📥 Enter or select a tweet to analyze sentiment")
tweet_input = st.text_area(" Type your tweet below:")

# Pre-filled sample tweets
sample_tweet = st.selectbox("Or try a sample tweet:", [
    "I love the flight experience!",
    "The flight was delayed and the staff was rude.",
    "It was okay, nothing special really."
])

# Auto-fill if input is blank
if tweet_input.strip() == "":
    tweet_input = sample_tweet

# Predict button
if st.button(" Predict Sentiment"):
    cleaned = clean_text(tweet_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    st.subheader("📊📈 Predicted Sentiment:")
    if prediction == "positive":
        st.success("😊 Positive — Great experience!")
    elif prediction == "neutral":
        st.info("😐 Neutral — Average or mixed opinion.")
    else:
        st.error("😡 Negative — Poor experience or complaint.")


# Visualizations from dataset

st.markdown("---")
st.header(" Sentiment Analysis on Full Dataset")

# Loading full dataset
df = pd.read_csv("Tweets.csv")

# Sentiment distribution chart
st.subheader("Sentiment Distribution (original dataset)")
fig, ax = plt.subplots()
sns.countplot(data=df, x="airline_sentiment", order=["positive", "neutral", "negative"], palette="Set2", ax=ax)
st.pyplot(fig)

# Word Cloud Section
st.subheader(" Word Clouds by Sentiment")

def plot_wordcloud(sentiment):
    text = " ".join(df[df.airline_sentiment == sentiment]["text"].astype(str))
    wc = WordCloud(width=600, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**😊 Positive**")
    st.pyplot(plot_wordcloud("positive"))

with col2:
    st.markdown("**😐 Neutral**")
    st.pyplot(plot_wordcloud("neutral"))

with col3:
    st.markdown("**😡 Negative**")
    st.pyplot(plot_wordcloud("negative"))


# Infomation Section

with st.expander("ℹ How it works"):
    st.markdown("""
    - This app uses **NLTK** for text preprocessing (stopwords, lemmatization).
    - It uses a **TF-IDF vectorizer** to convert text into numeric format.
    - A **Multinomial Naive Bayes model** predicts whether the tweet is Positive, Neutral, or Negative.
    - Word clouds and charts provide overall insights into public sentiment toward airlines.
    """)
