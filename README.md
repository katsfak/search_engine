# 🔍 Search Engine – Information Retrieval System  

This project implements a simple **Search Engine** as part of the **Information Retrieval** course.  
It demonstrates the full pipeline of **data collection, preprocessing, indexing, and retrieval** of academic documents using multiple search models.  

---

## ✨ Features  

- **Web Crawling**: Scrapes metadata (title, authors, abstract, date) from [Polynoe Repository (UNIWA)](https://polynoe.lib.uniwa.gr).  
- **Preprocessing**:  
  - Tokenization  
  - Stopword removal (Greek)  
  - Stemming & Lemmatization  
  - Removal of punctuation/special characters  
- **Indexing**: Builds an **inverted index** for efficient lookup.  
- **Retrieval Models**:  
  1. Boolean Retrieval (AND, OR, NOT)  
  2. Vector Space Model (**TF-IDF + Cosine Similarity**)  
  3. Probabilistic Model (**Okapi BM25**)  
- **Ranking**: Results are ranked by similarity score.  
- **Filtering (optional)**: Can filter by author or publication date.  
- **Interactive CLI**: User enters a query and selects the retrieval method.  

---

## 📂 Repository Structure  

```
search_engine/
├── main.py                # Main program: scraping + preprocessing + indexing + retrieval
├── data.json              # Raw scraped metadata
├── processed_data.json    # Preprocessed documents
├── inverted_index.json    # Inverted index
├── README.md              # Project documentation
```

---

## ⚙️ Installation  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/katsfak/search_engine.git
cd search_engine
```

If requirements library is missing, install manually:  

```bash
pip install requests beautifulsoup4 nltk scikit-learn rank-bm25 numpy
```

Download required **NLTK resources**:  

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

---

## ▶️ Usage  

Run the program:  

```bash
python main.py
```

Steps:  
1. The crawler collects metadata from Polynoe Repository.  
2. Documents are **preprocessed** and indexed.  
3. You enter a **search query**.  
4. Choose a retrieval model:  
   - `1` → Boolean Retrieval  
   - `2` → Vector Space Model (TF-IDF)  
   - `3` → Okapi BM25  
5. The system prints ranked search results with:  
   - Title  
   - Author  
   - Date  
   - Abstract  

---

## 🔎 Example  

```
Enter your search query: machine learning
Please choose an algorithm:
1. Boolean Retrieval
2. Vector Space Model
3. Okapi BM25
Enter your choice (1-3): 2

Similarity: 0.83
Title: deep machin learn appli
Author: john doe
Date: 2024-05-12
Abstract: studi deep learn techniqu optim perform...
```

---

## 📖 Lesson Context  

This project was developed as part of the **Information Retrieval** course, covering:  
- Web crawling & metadata extraction  
- Text preprocessing for retrieval  
- Indexing structures (inverted index)  
- Classical retrieval models (Boolean, VSM, BM25)  
- Ranking and evaluation of search results  

---

## 👨‍💻 Author  

Developed by **Katsfak** for the **Information Retrieval** course.
