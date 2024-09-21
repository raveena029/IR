import os
import math
import re
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Preprocessing function
def pre_processing_function(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    tokens = [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word) != '']
    return tokens

class VectorSpaceModel:
    def __init__(self):
        self.dictionary = defaultdict(list)
        self.doc_lengths = defaultdict(float)
        self.doc_count = 0

    def add_document(self, doc_id, text):
        tokens = pre_processing_function(text)
        term_freq = Counter(tokens)
        for term, freq in term_freq.items():
            self.dictionary[term].append((doc_id, freq))
        self.doc_lengths[doc_id] = math.sqrt(sum((1 + math.log10(freq))**2 for freq in term_freq.values()))
        self.doc_count += 1

    def calculate_idf(self, term):
        return math.log10(self.doc_count / len(self.dictionary[term]))

    def calculate_lnc(self, term, freq):
        return 1 + math.log10(freq)

    def calculate_ltc(self, term, freq, idf):
        return (1 + math.log10(freq)) * idf

    def rank_documents(self, query):
        query_terms = pre_processing_function(query)
        query_freq = Counter(query_terms)
        query_weights = {term: self.calculate_ltc(term, freq, self.calculate_idf(term)) for term, freq in query_freq.items()}
        query_length = math.sqrt(sum(weight**2 for weight in query_weights.values()))

        scores = defaultdict(float)
        for term, q_weight in query_weights.items():
            if term in self.dictionary:
                for doc_id, freq in self.dictionary[term]:
                    d_weight = self.calculate_lnc(term, freq)
                    scores[doc_id] += d_weight * q_weight

        for doc_id in scores:
            scores[doc_id] /= (self.doc_lengths[doc_id] * query_length)

        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_docs[:10]

def load_corpus(corpus_dir):
    vsm = VectorSpaceModel()
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                vsm.add_document(filename, text)
    return vsm

# Load the corpus
corpus_dir = r"C:\Users\ravee\Downloads\Corpus"
vsm = load_corpus(corpus_dir)

# Get the query from the user
query = input("Enter your query: ")

# Rank the documents based on the query
top_docs = vsm.rank_documents(query)

# Print the top documents
for rank, (doc_id, score) in enumerate(top_docs, start=1):
    print(f"{rank}. ('{doc_id}', {score})")
