# Task 2 – NLTK-Powered Text Analytics Web App

## 🔍 Overview

This project builds a natural language processing (NLP) tool using **NLTK**, **pandas**, and **Streamlit**. It analyzes text data with tokenization, POS tagging, word frequency, collocations, and sentiment scoring. A Streamlit-based web app lets users upload and analyze `.txt` files.

---

## 📁 Files

- `nlp_pipeline.py`: Core NLP functions using NLTK and pandas.
- `streamlit_app.py`: Web UI for interactive analysis.
- `train.ipynb`: Notebook for testing the pipeline locally.
- `example_input.txt`: A sample text file to test with.

---

## 🛠️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd Task2/


Download required NLTK models:

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

▶️ How to Run

streamlit run streamlit_app.py
📸 Required Screenshots
App UI after analysis

Word frequency chart

Tokenized or NER output display