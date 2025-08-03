# Task2/nlp_pipeline.py

import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, FreqDist, collocations
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return tokens

def analyze_text(text):
    tokens = preprocess_text(text)
    pos_tags = pos_tag(tokens)
    freq_dist = FreqDist(tokens)
    collocation_finder = nltk.BigramCollocationFinder.from_words(tokens)
    collocations_top = collocation_finder.nbest(nltk.BigramAssocMeasures().pmi, 5)

    sentiment = TextBlob(text).sentiment

    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "top_words": freq_dist.most_common(10),
        "collocations": collocations_top,
        "sentiment": {
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity
        }
    }

def analyze_dataframe(df, text_column):
    results = []
    for row in df[text_column]:
        result = analyze_text(row)
        results.append(result)
    return results
