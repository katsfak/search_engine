# Βιβλιοθήκες
import math
import requests
from flask import Flask, request, render_template
from bs4 import BeautifulSoup
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import string
import numpy as np


# Ερώτημα 1
# a Επιλέξτε έναν ιστότοπο-στόχο ή ένα αποθετήριο ακαδημαϊκών εργασιών (π.χ. arXiv, PubMed ή αποθετήριο πανεπιστημίου). 
def scrape_polynoe():
    url = 'https://polynoe.lib.uniwa.gr/xmlui/browse?type=dateissued'
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    descriptions = soup.find_all('div', class_='artifact-description')
    data = []

    for desc in descriptions:
        title = desc.find('h4', class_='artifact-title').text.strip()
        author = desc.find('span', class_='author h4').text.strip()
        date = desc.find('span', class_='date').text.strip()
        abstract = desc.find('div', class_='artifact-abstract').text.strip()
        data.append([title, author, date, abstract])

    with open('data.json', 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)

# Καλέστε τη συνάρτηση
scrape_polynoe()


# Ερώτημα 2
# Κάντε προεπεξεργασία του κειμενικού περιεχομένου των ακαδημαϊκών εργασιών για την προετοιμασία τους για ευρετηρίαση και αναζήτηση. Αυτό μπορεί να περιλαμβάνει εργασίες όπως tokenization, stemming/lemmatization και stop-word removal και αφαίρεση ειδικών χαρακτήρων (removing special characters).
# Φορτώστε τις stop words
def preprocess_text():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('greek'))  # Define stop_words
    stemmer = PorterStemmer()  # Define stemmer

    with open('data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    processed_data = []
    for entry in data:
        title_tokens = word_tokenize(entry['title'])
        title_tokens = [word.lower() for word in title_tokens if word.isalpha() and word.lower() not in stop_words]
        title_tokens = [stemmer.stem(word) for word in title_tokens]
        author_tokens = word_tokenize(entry['author'])
        author_tokens = [word.lower() for word in author_tokens if word.isalpha() and word.lower() not in stop_words]
        author_tokens = [stemmer.stem(word) for word in author_tokens]
        abstract_tokens = word_tokenize(entry['abstract'])
        abstract_tokens = [word.lower() for word in abstract_tokens if word.isalpha() and word.lower() not in stop_words]
        abstract_tokens = [stemmer.stem(word) for word in abstract_tokens]

        processed_data.append({
            'title': title_tokens,
            'author': author_tokens,
            'date': entry['date'],
            'abstract': abstract_tokens
        })
    with open('processed_data.json', 'w', encoding='utf8') as f:
        json.dump(processed_data, f, ensure_ascii=False)

preprocess_text()

# Ερώτημα 3
# α. Δημιουργήστε μια ανεστραμμένη δομή δεδομένων ευρετηρίου (inverted index) για την αποτελεσματική αντιστοίχιση όρων στα έγγραφα στα οποία εμφανίζονται. 
# Δημιουργία ενός ανεστραμμένου ευρετηρίου
def create_inverted_index():
    with open('processed_data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    inverted_index = defaultdict(set)
    for i, entry in enumerate(data):
        for word in entry['title']:
            inverted_index[word].add(i)
        for word in entry['author']:
            inverted_index[word].add(i)
        for word in entry['abstract']:
            inverted_index[word].add(i)
    inverted_index = {k: list(v) for k, v in inverted_index.items()}
    with open('inverted_index.json', 'w', encoding='utf8') as f:
        json.dump(inverted_index, f, ensure_ascii=False)

create_inverted_index()

# Ερώτημα 4
# α. Αναπτύξτε μια διεπαφή χρήστη για την αναζήτηση ακαδημαϊκών εργασιών χρησιμοποιώντας την Python (π.χ. μια διεπαφή γραμμής εντολών ή μια απλή διεπαφή ιστού). 
