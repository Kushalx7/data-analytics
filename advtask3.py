import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
from wordcloud import WordCloud


df = pd.read_csv("D:/kagglee/apple_jobs.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('&', '_')
print(df.columns)
df['preferred_qual'] = df['preferred_qual'].fillna('')
df['education_experience'] = df['education_experience'].fillna('')
print(df.isnull().sum())

#selecting text column
df['text'] = df['responsibilities']
# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text)

    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words and word not in string.punctuation:
            cleaned_tokens.append(stemmer.stem(word))

    return " ".join(cleaned_tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

#sentiment analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['clean_text'].apply(get_sentiment)

print(df[['text', 'clean_text', 'sentiment']].head())
print(df['sentiment'].value_counts())
#Sentiment Distribution Bar Chart
df['sentiment'].value_counts().plot(kind='bar')

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Records")
plt.show()
#word cloud
all_words = " ".join(df['clean_text'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud of Job Responsibilities")
plt.show()

#Preprocessing	Clean text
# Sentiment analysis Understand tone
#Distribution Overall pattern
#word cloud key topics
