# Βιβλιοθήκες
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
from collections import defaultdict


# Ερώτημα 1
url = 'https://polynoe.lib.uniwa.gr/xmlui/browse?type=dateissued'
html = requests.get(url)
soup = BeautifulSoup(html.content, 'html.parser')

# Βρείτε όλα τα divs με την κλάση 'artifact-description'
descriptions = soup.find_all('div', class_='artifact-description')

# Δημιουργία λίστας για την αποθήκευση των δεδομένων
data = []

# Εκτύπωση του κειμένου κάθε τίτλου εργασίας
for desc in descriptions:
    title = desc.find('h4', class_='artifact-title').text.strip()
    author = desc.find('span', class_='author h4').text.strip()
    date = desc.find('span', class_='date').text.strip()
    abstract = desc.find('div', class_='artifact-abstract').text.strip()
    data.append([title, author, date, abstract])

# Αποθήκευση των δεδομένων σε μορφή JSON
with open('data.json', 'w', encoding='utf8') as f:
    json.dump(data, f, ensure_ascii=False)


# Ερώτημα 2
# Φορτώστε τις stop words
stop_words = set(stopwords.words('greek'))

# Κατασκευή αντικειμένου για την κορένωση (stemming)
stemmer = PorterStemmer()

# Προεπεξεργασία των δεδομένων
processed_data = []

for entry in data:
    title_tokens = word_tokenize(entry[0])
    title_tokens = [word.lower() for word in title_tokens if word.isalpha() and word.lower() not in stop_words]
    title_tokens = [stemmer.stem(word) for word in title_tokens]

    author_tokens = word_tokenize(entry[1])
    author_tokens = [word.lower() for word in author_tokens if word.isalpha() and word.lower() not in stop_words]
    author_tokens = [stemmer.stem(word) for word in author_tokens]

    abstract_tokens = word_tokenize(entry[3])
    abstract_tokens = [word.lower() for word in abstract_tokens if word.isalpha() and word.lower() not in stop_words]
    abstract_tokens = [stemmer.stem(word) for word in abstract_tokens]

    processed_data.append({
        'title': title_tokens,
        'author': author_tokens,
        'date': entry[2],
        'abstract': abstract_tokens
    })

# Αποθήκευση των επεξεργασμένων δεδομένων σε μορφή JSON
with open('processed_data.json', 'w', encoding='utf8') as f:
    json.dump(processed_data, f, ensure_ascii=False)

# Ερώτημα 3
# Δημιουργία ενός ανεστραμμένου ευρετηρίου
inverted_index = defaultdict(set)

for i, entry in enumerate(processed_data):
    for word in entry['title']:
        inverted_index[word].add(i)
    for word in entry['author']:
        inverted_index[word].add(i)
    for word in entry['abstract']:
        inverted_index[word].add(i)

# Convert the sets to lists
inverted_index = {k: list(v) for k, v in inverted_index.items()}

# Αποθήκευση του ανεστραμμένου ευρετηρίου σε μορφή JSON
with open('inverted_index.json', 'w', encoding='utf8') as f:
    json.dump(inverted_index, f, ensure_ascii=False)

# Ερώτημα 4
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        with open('inverted_index.json', 'r', encoding='utf8') as f:
            inverted_index = json.load(f)
        with open('processed_data.json', 'r', encoding='utf8') as f:
            processed_data = json.load(f)
        for word in query.split():
            if word in inverted_index:
                for i in inverted_index[word]:
                    results.append(processed_data[i])
    return render_template('search.html', results=results)
# Υποθέτουμε ότι έχουμε τα επεξεργασμένα δεδομένα από το προηγούμενο παράδειγμα
processed_data = [
    {'title': ['sample', 'title'], 'author': ['sample', 'author'], 'date': '2022-01-01', 'abstract': ['sample', 'abstract']},
    # Προσθέστε τα υπόλοιπα επεξεργασμένα δεδομένα εδώ...
]

# Λίστα με τα ερωτήματα των χρηστών
user_queries = ["sample query 1", "sample query 2"]

# Υλοποίηση του αλγορίθμου Boolean retrieval
def boolean_retrieval(query, data):
    results = []
    for idx, entry in enumerate(data):
        if any(term in entry['title'] or term in entry['abstract'] for term in query.split()):
            results.append(idx)
    return results

# Υλοποίηση του αλγορίθμου Vector Space Model (VSM)
def vsm_retrieval(query, data):
    vectorizer = TfidfVectorizer()
    documents = [' '.join(entry['title'] + entry['abstract']) for entry in data]
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    results = np.argsort(cosine_similarities)[::-1]
    return results

# Υλοποίηση του αλγορίθμου Okapi BM25
def bm25_retrieval(query, data):
    tokenized_data = [' '.join(entry['title'] + entry['abstract']) for entry in data]
    tokenized_query = query.split()
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tokenized_data)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(X)

    bm25 = BM25Okapi(tokenized_data)
    scores = bm25.get_scores(tokenized_query)

    results = np.argsort(scores)[::-1]
    return results

# Εφαρμογή των αλγορίθμων για κάθε ερώτημα του χρήστη
for query in user_queries:
    print(f"\nUser Query: {query}")
    
    # Boolean retrieval
    boolean_results = boolean_retrieval(query, processed_data)
    print(f"Boolean Retrieval Results: {boolean_results}")

    # Vector Space Model (VSM)
    vsm_results = vsm_retrieval(query, processed_data)
    print(f"VSM Results: {vsm_results}")

    # Okapi BM25
    bm25_results = bm25_retrieval(query, processed_data)
    print(f"BM25 Results: {bm25_results}")







