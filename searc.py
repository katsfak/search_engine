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
url = 'https://polynoe.lib.uniwa.gr/xmlui/browse?type=dateissued'
html = requests.get(url)
soup = BeautifulSoup(html.content, 'html.parser')

# β. Υλοποιήστε έναν web crawler σε Python (π.χ. με BeautifulSoup) για τη συλλογή μεταδεδομένων ακαδημαϊκών εργασιών (τίτλος, συγγραφείς, περίληψη, ημερομηνία δημοσίευσης κ.λπ.) από την επιλεγμένη πηγή.
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
# γ. Αποθηκεύστε τα δεδομένα που συλλέγονται σε δομημένη μορφή, όπως JSON ή CSV.
with open('data.json', 'w', encoding='utf8') as f:
    json.dump(data, f, ensure_ascii=False)


# Ερώτημα 2
# Κάντε προεπεξεργασία του κειμενικού περιεχομένου των ακαδημαϊκών εργασιών για την προετοιμασία τους για ευρετηρίαση και αναζήτηση. Αυτό μπορεί να περιλαμβάνει εργασίες όπως tokenization, stemming/lemmatization και stop-word removal και αφαίρεση ειδικών χαρακτήρων (removing special characters).
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
# α. Δημιουργήστε μια ανεστραμμένη δομή δεδομένων ευρετηρίου (inverted index) για την αποτελεσματική αντιστοίχιση όρων στα έγγραφα στα οποία εμφανίζονται. 
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

# β Εφαρμόστε μια δομή δεδομένων για την αποθήκευση της αντιστοίχισης μεταξύ λέξεων κλειδιών (keywords) και εγγράφων.
# Αποθήκευση του ανεστραμμένου ευρετηρίου σε μορφή JSON
with open('inverted_index.json', 'w', encoding='utf8') as f:
    json.dump(inverted_index, f, ensure_ascii=False)

# Ερώτημα 4
# α. Αναπτύξτε μια διεπαφή χρήστη για την αναζήτηση ακαδημαϊκών εργασιών χρησιμοποιώντας την Python (π.χ. μια διεπαφή γραμμής εντολών ή μια απλή διεπαφή ιστού). 
app = Flask(__name__ , template_folder='templates' )

# Φορτώστε τα δεδομένα και τον αντίστροφο ευρετήριο από τα αρχεία JSON
with open('data.json', 'r', encoding='utf8') as f:
    data = json.load(f)

with open('inverted_index.json', 'r', encoding='utf8') as f:
    inverted_index = json.load(f)

# Υλοποίηση του Boolean retrieval algorithm
def boolean_retrieval(query, inverted_index):
    """
    Perform a boolean retrieval operation.
    
    Parameters:
    query (str): The query terms.
    inverted_index (dict): The inverted index data structure.

    Returns:
    list: A list of document IDs that match the query.
    """
    # Initialize the result set with all available documents
    # Use the size of the data, not the index
    results = set(range(len(data)))  

    # Split the query into terms
    for term in query.split():  
        if term in inverted_index:
            results &= set(inverted_index[term])
        else:
            # If a term is not in the index, the result is an empty set
            results = set()
            break

    return list(results)


# Υλοποίηση του Vector Space Model (VSM)
def vector_space_model(query, inverted_index, document_vectors):
    # Υπολογισμός του σκορ συνάφειας για κάθε έγγραφο
    scores = defaultdict(float)
    for term in query.split():  # Χωρίστε το query σε λέξεις
        if term in inverted_index:
            doc_ids = inverted_index[term]
            for doc_id in doc_ids:
                scores[doc_id] += document_vectors[str(doc_id)].get(term, 0)  # Μετατρέψτε το doc_id σε string
    # Κατάταξη των εγγράφων με βάση το σκορ συνάφειας
    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in ranked_results]

# Υλοποίηση του Probabilistic retrieval model (π.χ. Okapi BM25)
def probabilistic_retrieval(query, inverted_index, document_vectors, k1=1.5, b=0.75):
    # Υπολογισμός του μέσου μήκους των εγγράφων
    avg_doc_length = sum(len(document_vectors[str(doc_id)]) for doc_id in document_vectors) / len(document_vectors)
    # Υπολογισμός του σκορ BM25 για κάθε έγγραφο
    scores = defaultdict(float)
    for term in query.split():  # Χωρίστε το query σε λέξεις
        if term in inverted_index:
            df = len(inverted_index[term])
            for doc_id in inverted_index[term]:
                tf = document_vectors[str(doc_id)].get(term, 0)  # Μετατρέψτε το doc_id σε string
                doc_length = len(document_vectors[str(doc_id)])
                # Υπολογισμός των στοιχείων του σκορ BM25
                idf = math.log((len(document_vectors) - df + 0.5) / (df + 0.5))
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
                scores[doc_id] += idf * numerator / denominator
    # Κατάταξη των εγγράφων με βάση το σκορ BM25
    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in ranked_results]

# Αναπτύξτε τη διεπαφή χρήστη
@app.route('/', methods=['GET', 'POST'])
def search_interface():
    if request.method == 'POST':
        query = request.form.get('query')
        filter_date = request.form.get('filter_date')
        filter_author = request.form.get('filter_author')

        # Perform the search based on the query and filters
        try:
            results = boolean_retrieval(query, inverted_index)  # Add the inverted_index
        except Exception as e:
            return render_template('error.html', error=str(e))

        # Filter the results based on the date and author
        if filter_date or filter_author:
            results = filter_results(results, filter_date, filter_author)

        return render_template('results.html', results=results)
    else:
        return render_template('search.html')


def filter_results(results, filter_date, filter_author):
    filtered_results = []
    for r in results:
        doc = data[str(r)]  # Use the id to get the document from the data
        if filter_date and doc['date'] != filter_date:
            continue
        if filter_author and filter_author.lower() not in doc['author'].lower():
            continue
        filtered_results.append(doc)
    return filtered_results


if __name__ == '__main__':
    app.run(debug=True)

