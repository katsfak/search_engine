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
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import json
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

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
    nltk.download('wordnet')

    stop_words = set(stopwords.words('greek'))  # Define stop_words
    stemmer = PorterStemmer()  # Define stemmer
    lemmatizer = WordNetLemmatizer()  # Define lemmatizer

    with open('data.json', 'r', encoding='utf8') as f:
        data = json.load(f)

    processed_data = []
    for entry in data:
        title_tokens = word_tokenize(entry[0])
        title_tokens = [word.lower() for word in title_tokens if word.isalpha() and word.lower() not in stop_words]
        title_tokens = [stemmer.stem(word) for word in title_tokens]
        title_tokens = [lemmatizer.lemmatize(word) for word in title_tokens]

        author_tokens = word_tokenize(entry[1])
        author_tokens = [word.lower() for word in author_tokens if word.isalpha() and word.lower() not in stop_words]
        author_tokens = [stemmer.stem(word) for word in author_tokens]
        author_tokens = [lemmatizer.lemmatize(word) for word in author_tokens]

        abstract_tokens = word_tokenize(entry[3])
        abstract_tokens = [word.lower() for word in abstract_tokens if word.isalpha() and word.lower() not in stop_words]
        abstract_tokens = [stemmer.stem(word) for word in abstract_tokens]
        abstract_tokens = [lemmatizer.lemmatize(word) for word in abstract_tokens]

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

def search():
    print("Please choose an algorithm:")
    print("1. Boolean Retrieval")
    print("2. Vector Space Model")
    print("3. Okapi BM25")
    choice = int(input("Enter your choice (1-3): "))

    # Load the documents from the JSON file
    with open('processed_data.json', 'r', encoding='utf8') as f:
        documents = json.load(f)

    if choice == 1:
        title = input("Παρακαλώ εισάγετε τον τίτλο: ")
        result = query_processing(title)
    elif choice == 2:
        title = input("Παρακαλώ εισάγετε τον τίτλο: ")
        result = vector_space_model(title)
    elif choice == 3:
        title = input("Παρακαλώ εισάγετε τον τίτλο: ")
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
    stack = []
    operators = set(['AND', 'OR', 'NOT'])
    
    for token in query_tokens:
        if token not in operators:
            # If the token is not an operator, retrieve the relevant documents
            stack.append(set(inverted_index.get(token, [])))
        else:
            # If the token is an operator, perform the corresponding operation
            if token == 'AND':
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = operand1.intersection(operand2)
                stack.append(result)
            elif token == 'OR':
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = operand1.union(operand2)
                stack.append(result)
            elif token == 'NOT':
                operand = stack.pop()
                all_docs = set(inverted_index.keys())
                result = all_docs.difference(operand)
                stack.append(result)

    # Print the final result if it's on top of the stack
    if stack:
        final_result = list(stack[0])
        for idx in final_result:
            doc = documents[idx]
            print(f"Title: {doc['title']}\nAuthor: {doc['author']}\nDate: {doc['date']}\nAbstract: {doc['abstract']}\n")
    else:
        print("No results found.")

def vector_space_model(query):
    # Load preprocessed documents from JSON file
    with open('processed_data.json', 'r', encoding='utf8') as f:
        documents = json.load(f)

    # Tokenize the query
    tokenized_query = word_tokenize(query.lower())

    # Calculate TF-IDF
    # Convert tokenized documents to text
    preprocessed_documents = [' '.join(doc['title'] + doc['author'] + doc['abstract'] + [doc['date']]) for doc in documents]  # Combine all fields
    preprocessed_query = ' '.join(tokenized_query)

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

    # Transform the query into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Rank documents by similarity
    results = [(documents[i], cosine_similarities[0][i]) for i in range(len(documents))]
    results.sort(key=lambda x: x[1], reverse=True)

    # Print the top 5 ranked documents
    for doc, similarity in results[:5]:  
        print(f"Similarity: {similarity:.2f}\nTitle: {' '.join(doc['title'])}\nAuthor: {' '.join(doc['author'])}\nDate: {doc['date']}\nAbstract: {' '.join(doc['abstract'])}\n")  # Print all fields

def okapibm25(query):
    # Load preprocessed documents from JSON file
    with open('processed_data.json', 'r', encoding='utf8') as f:
        documents = json.load(f)

    # Tokenize the query
    tokenized_query = query.split(" ")

    # Convert tokenized documents to text
    preprocessed_documents = [' '.join(doc['title'] + doc['author'] + doc['abstract'] + [doc['date']]) for doc in documents]  # Combine all fields

    # Initialize BM25Okapi model
    bm25 = BM25Okapi([doc.split(" ") for doc in preprocessed_documents])

    # Get scores for each document
    doc_scores = bm25.get_scores(tokenized_query)

    # Get the indices of the top documents
    top_indices = bm25.get_top_n(tokenized_query, range(len(preprocessed_documents)), n=5)

    # Print the details of the top documents
    for index in top_indices:
        print(f"Similarity Score: {doc_scores[index]}")
        print(f"Title: {documents[index]['title']}")
        print(f"Author: {documents[index]['author']}")
        print(f"Abstract: {documents[index]['abstract']}")
        print(f"Date: {documents[index]['date']}")
        print("\n")


    
# γ. Επιτρέψτε στους χρήστες να φιλτράρουν τα αποτελέσματα αναζήτησης με διάφορα 
# κριτήρια, όπως η ημερομηνία δημοσίευσης ή ο συγγραφέας.

def filter_results(criteria, value):
    # Άνοιγμα του αρχείου με τα επεξεργασμένα δεδομένα
    with open('processed_data.json', 'r', encoding='utf8') as f:
        data = json.load(f)

    # Δημιουργία μιας λίστας με τα έγγραφα που πληρούν το κριτήριο
    filtered_data = [doc for doc in data if doc.get(criteria) == value]
    print(filtered_data)

    # Επιστροφή της λίστας με τα φιλτραρισμένα δεδομένα
    return filtered_data

# Επεξεργασία ερωτήματος (Query Processing): Αναπτύξτε ένα module επεξεργασίας 
# ερωτημάτων που θα προεπεξεργάζεται τα ερωτήματα που λαμβάνει από τον χρήστη, τα αναλύει 
# και ανακτά σχετικά έγγραφα χρησιμοποιώντας το ανεστραμμένο ευρετήριο. Μπορείτε να 
# χρησιμοποιήσετε απλά ερωτήματα βάσει λέξεων (όρων). Οι χρήστες θα πρέπει να μπορούν να 
# αναζητούν έγγραφα χρησιμοποιώντας μία ή περισσότερες λέξεις. Το module θα λαμβάνει 
# ερωτήματα χρηστών τα οποία τα γίνονται tokenized και θα εκτελεί λειτουργίες Boolean (AND, OR
# και NOT). 

def query_processing(query,criteria):
    criteria = criteria
    # Άνοιγμα του αρχείου με το ανεστραμμένο ευρετήριο
    with open('inverted_index.json', 'r', encoding='utf8') as f:
        inverted_index = json.load(f)

    # Διαχωρισμός του ερωτήματος σε tokens
    query_tokens = query.split()

    # Αρχικοποίηση του συνόλου των σχετικών εγγράφων
    relevant_docs = set(inverted_index.get(query_tokens[0], []))

    # Αρχικοποίηση της λίστας των φιλτραρισμένων δεδομένων
    filtered_data = []
    # Επεξεργασία των υπόλοιπων tokens του ερωτήματος
    for token in enumerate(query_tokens[1:]):
        if token.upper() == 'AND' or token.upper() == 'and':
            continue
        elif token.upper() == 'OR' or token.upper() == 'or':
            # Ένωση των σχετικών εγγράφων με τα έγγραφα που αντιστοιχούν στο επόμενο token
            relevant_docs = relevant_docs.union(inverted_index.get(query_tokens[token+1], []))
        elif token.upper() == 'NOT' or token.upper() == 'not':
            # Διαφορά των σχετικών εγγράφων με τα έγγραφα που αντιστοιχούν στο επόμενο token
            relevant_docs = relevant_docs.difference(inverted_index.get(query_tokens[token+1], []))
        else:
            # Τομή των σχετικών εγγράφων με τα έγγραφα που αντιστοιχούν στο token
            relevant_docs = relevant_docs.intersection(inverted_index.get(token, []))

    # Επιστροφή της λίστας των σχετικών εγγράφων
    relevant_docs = list(relevant_docs)

    # Κλήση της συνάρτησης filter_results για κάθε σχετικό έγγραφο
    for doc in relevant_docs:
        filtered_data = filter_results(criteria, doc)

    return filtered_data

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
    scrape_polynoe()
    preprocess_text()
    create_inverted_index()
    search_query = input("Enter your search query: ")
    filters = input("Enter your filter: ")
    print(okapibm25(search_query))

