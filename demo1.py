# MACHINE LEARNING PROJECT IMPLEMENTATION
# AirSentiment Analysis (Airline Tweet Sentiment Predictor)
import pandas as pd
import nltk
import re
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.utils import resample
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloading NLTK Data 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Loading and Prepare Dataset

df = pd.read_csv("Tweets.csv")
df = df[['text', 'airline_sentiment']]  


#  Balancing the Dataset

negative = df[df.airline_sentiment == 'negative']
positive = df[df.airline_sentiment == 'positive']
neutral  = df[df.airline_sentiment == 'neutral']

# Upsampling to balance all classes
positive_up = resample(positive, replace=True, n_samples=len(negative), random_state=42)
neutral_up  = resample(neutral,  replace=True, n_samples=len(negative), random_state=42)

# Combine and shuffle
df_balanced = pd.concat([negative, positive_up, neutral_up])
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Preprocess Tweets

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Cleaning all tweets
df_balanced['clean_text'] = df_balanced['text'].apply(clean_text)


X = df_balanced['clean_text']
y = df_balanced['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Vectorization

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

#Training Naive Bayes Model

model = MultinomialNB()
model.fit(X_train_vec, y_train)


# Evaluating Model

y_pred = model.predict(X_test_vec)
print("\n Model Evaluation Report:\n")
print(classification_report(y_test, y_pred))


# Saving Model and Vectorizer

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n Model and vectorizer saved as 'sentiment_model.pkl' and 'vectorizer.pkl'")


#  Generating and Showing Word Clouds


def generate_wordcloud(sentiment, color):
    text = " ".join(df[df['airline_sentiment'] == sentiment]['text'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white', colormap=color).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"{sentiment.capitalize()} Word Cloud", fontsize=16)
    plt.show()

print("\n Generating Word Clouds...")
generate_wordcloud("positive", "Greens")
generate_wordcloud("neutral", "Blues")
generate_wordcloud("negative", "Reds")
print(" Word clouds displayed.")
