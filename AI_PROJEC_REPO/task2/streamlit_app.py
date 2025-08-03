import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import FreqDist, pos_tag, word_tokenize
from wordcloud import WordCloud
from textblob import TextBlob
import io

from nlp_pipeline import preprocess_text

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

st.set_page_config(page_title="Text Analytics App", layout="wide")

# Sidebar Routing
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Explorer", "Analysis Dashboard"])

# App-wide state
if "df" not in st.session_state:
    st.session_state.df = None

# Upload Section
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")
    preprocessed, tokens = preprocess_text(raw_text)
    df = pd.DataFrame(tokens, columns=["tokens"])
    st.session_state.df = df
    st.session_state.raw_text = raw_text
    st.session_state.preprocessed_text = preprocessed

# Page: Data Explorer
if page == "Data Explorer":
    st.title("📊 Data Explorer")

    if st.session_state.df is not None:
        st.subheader("Raw Text Sample")
        st.text_area("Original Text", st.session_state.raw_text[:1000], height=200)

        st.subheader("Cleaned Tokens")
        st.write(st.session_state.df.head())

        st.subheader("Top Words Frequency")
        freq_dist = FreqDist(st.session_state.df["tokens"])
        top_words = freq_dist.most_common(20)
        freq_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

        st.bar_chart(freq_df.set_index("Word"))

        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=300, background_color='white').generate(" ".join(st.session_state.df["tokens"]))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt.gcf())
    else:
        st.warning("Please upload a text file to explore the data.")

# Page: Analysis Dashboard
elif page == "Analysis Dashboard":
    st.title("📈 Analysis Dashboard")

    if st.session_state.df is not None:
        st.subheader("POS Tag Distribution")
        pos_tags = pos_tag(st.session_state.df["tokens"])
        pos_counts = pd.Series([tag for word, tag in pos_tags]).value_counts().head(10)
        st.bar_chart(pos_counts)

        st.subheader("Sentiment Analysis")
        blob = TextBlob(st.session_state.preprocessed_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        st.metric("Polarity", f"{polarity:.2f}", help="[-1, +1] Negative to Positive")
        st.metric("Subjectivity", f"{subjectivity:.2f}", help="[0, 1] Objective to Subjective")

        st.progress((polarity + 1) / 2)
    else:
        st.warning("Please upload a text file to run analysis.")
