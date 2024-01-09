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
# β. Υλοποιήστε έναν web crawler σε Python (π.χ. με BeautifulSoup) για τη συλλογή 
# μεταδεδομένων ακαδημαϊκών εργασιών (τίτλος, συγγραφείς, περίληψη, ημερομηνία 
# δημοσίευσης κ.λπ.) από την επιλεγμένη πηγή.
# γ. Αποθηκεύστε τα δεδομένα που συλλέγονται σε δομημένη μορφή, όπως JSON ή CSV.

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

# Ερώτημα 2
# Κάντε προεπεξεργασία του κειμενικού περιεχομένου των ακαδημαϊκών εργασιών για την προετοιμασία τους για ευρετηρίαση και αναζήτηση. 
# Αυτό μπορεί να περιλαμβάνει εργασίες όπως tokenization, stemming/lemmatization και stop-word removal και αφαίρεση ειδικών χαρακτήρων 
# (removing special characters).
        
def preprocess_text():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('greek'))  # Define stop_words
    stemmer = PorterStemmer()  # Define stemmer

    with open('data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
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
    with open('processed_data.json', 'w', encoding='utf8') as f:
        json.dump(processed_data, f, ensure_ascii=False)

# Ερώτημα 3
# α. Δημιουργήστε μια ανεστραμμένη δομή δεδομένων ευρετηρίου (inverted index) για την αποτελεσματική αντιστοίχιση όρων
#  στα έγγραφα στα οποία εμφανίζονται. 
# β. Εφαρμόστε μια δομή δεδομένων για την αποθήκευση του ευρετηρίου.
        
def create_inverted_index():
    with open('processed_data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    inverted_index = defaultdict(set)
    for i, entry in enumerate(data):
        for word in entry['abstract']:
            inverted_index[word].add(i)
    inverted_index = {k: list(v) for k, v in inverted_index.items()}
    with open('inverted_index.json', 'w', encoding='utf8') as f:
        json.dump(inverted_index, f, ensure_ascii=False)


# Ερώτημα 4
# α. Αναπτύξτε μια διεπαφή χρήστη για την αναζήτηση ακαδημαϊκών εργασιών χρησιμοποιώντας την Python (π.χ. μια διεπαφή γραμμής εντολών ή μια απλή διεπαφή ιστού). 

def search(title):
    print("Please choose an algorithm:")
    print("1. Boolean Retrieval")
    print("2. Vector Space Model")
    print("3. Okapi BM25")
    choice = int(input("Enter your choice (1-3): "))

    # Load the documents from the JSON file
    with open('processed_data.json', 'r', encoding='utf8') as f:
        documents = json.load(f)

    if choice == 1:
        result = boolean_retrieval(title, documents)
    elif choice == 2:
        result = vector_space_model(documents, title)
    elif choice == 3:
        result = okapibm25()
    else:
        print("Invalid choice. Please enter a number between 1 and 3.")
        return

    print("The result of your search is:")
    print(result)

# β. Υλοποιήστε πολλαπλούς (τουλάχιστον 3) αλγόριθμους ανάκτησης, όπως Boolean retrieval, 
# Vector Space Model (VSM) και Probabilistic retrieval models (π.χ. Okapi BM25) για να 
# ανακτήσετε σχετικές εργασίες με βάση τα ερωτήματα των χρηστών. Ο χρήστης θα μπορεί 
# να επιλέγει τον αλγόριθμο ανάκτησης.
        
def boolean_retrieval(query):
    with open('inverted_index.json', 'r', encoding='utf8') as f:
        inverted_index = json.load(f)
    query_tokens = query.split()
    relevant_docs = set(inverted_index.get(query_tokens[0], []))
    for token in query_tokens[1:]:
        relevant_docs.intersection_update(inverted_index.get(token, []))
    return list(relevant_docs)

def vector_space_model(query, documents):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)
    return similarities[0]

def okapibm25(k1=1.5, b=0.75):
    with open('processed_data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    documents = [' '.join(doc['abstract']) for doc in data]
    vectorizer = CountVectorizer()
    doc_vectors = vectorizer.fit_transform(documents).toarray()
    avgdl = np.mean([len(doc) for doc in documents])
    idf = np.log((len(documents) - np.count_nonzero(doc_vectors, axis=0) + 0.5) / (np.count_nonzero(doc_vectors, axis=0) + 0.5))
    def score(query):
        query_vector = vectorizer.transform([query]).toarray()[0]
        dl = len(query.split())
        tf = query_vector / (1 - b + b * dl / avgdl)
        return np.sum(idf * tf * (k1 + 1) / (tf + k1), axis=1)
    return score

# γ. Επιτρέψτε στους χρήστες να φιλτράρουν τα αποτελέσματα αναζήτησης με διάφορα 
# κριτήρια, όπως η ημερομηνία δημοσίευσης ή ο συγγραφέας.

def filter_results(criteria, value):
    with open('processed_data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    filtered_data = [doc for doc in data if doc[criteria] == value]
    return filtered_data

# Επεξεργασία ερωτήματος (Query Processing): Αναπτύξτε ένα module επεξεργασίας 
# ερωτημάτων που θα προεπεξεργάζεται τα ερωτήματα που λαμβάνει από τον χρήστη, τα αναλύει 
# και ανακτά σχετικά έγγραφα χρησιμοποιώντας το ανεστραμμένο ευρετήριο. Μπορείτε να 
# χρησιμοποιήσετε απλά ερωτήματα βάσει λέξεων (όρων). Οι χρήστες θα πρέπει να μπορούν να 
# αναζητούν έγγραφα χρησιμοποιώντας μία ή περισσότερες λέξεις. Το module θα λαμβάνει 
# ερωτήματα χρηστών τα οποία τα γίνονται tokenized και θα εκτελεί λειτουργίες Boolean (AND, OR
# και NOT). 

def query_processing(query):
    with open('inverted_index.json', 'r', encoding='utf8') as f:
        inverted_index = json.load(f)
    query_tokens = query.split()
    relevant_docs = set(inverted_index.get(query_tokens[0], []))
    for token in query_tokens[1:]:
        if token.upper() == 'AND':
            continue
        elif token.upper() == 'OR':
            relevant_docs = relevant_docs.union(inverted_index.get(query_tokens[i+1], []))
        elif token.upper() == 'NOT':
            relevant_docs = relevant_docs.difference(inverted_index.get(query_tokens[i+1], []))
        else:
            relevant_docs = relevant_docs.intersection(inverted_index.get(token, []))
    return list(relevant_docs)

# Κατάταξη αποτελεσμάτων (Ranking): Εφαρμόστε έναν βασικό αλγόριθμο κατάταξης. Μπορείτε 
# να ξεκινήσετε με έναν απλό αλγόριθμο κατάταξης TF-IDF (Term Frequency-Inverse Document
# Frequency) και αργότερα μπορείτε να συμπεριλάβετε πιο προηγμένες τεχνικές κατάταξης. 
# Ταξινομήστε και παρουσιάστε τα αποτελέσματα αναζήτησης σε φιλική προς το χρήστη μορφή.

def rank_results(query, documents):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    tfidf_scores = np.dot(doc_vectors, query_vector.T).toarray()
    ranked_docs = np.argsort(-tfidf_scores, axis=0)
    return ranked_docs

if __name__ == "__main__":
    title = input("Παρακαλώ εισάγετε τον τίτλο: ")
    scrape_polynoe()
    preprocess_text()
    create_inverted_index()
    #search()