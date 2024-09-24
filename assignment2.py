# import os
# import math
# import re
# from collections import defaultdict, Counter
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

# def pre_processing_function(text):
#     text = text.lower()
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
#     ps = PorterStemmer()
#     tokens = [ps.stem(word) for word in tokens]
#     tokens = [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word) != '']
#     return tokens if tokens else ['placeholder']

# class VectorSpaceModel:
#     def __init__(self):
#         self.dictionary = defaultdict(list)  # term -> [(doc_id, tf)]
#         self.doc_lengths = defaultdict(float)  # doc_id -> length
#         self.doc_count = 0

#     def add_document(self, doc_id, text):
#         tokens = pre_processing_function(text)
#         term_freq = Counter(tokens)
#         for term, freq in term_freq.items():
#             self.dictionary[term].append((doc_id, freq))
#         self.doc_count += 1

#     def calculate_idf(self, term):
#         df = len(self.dictionary[term])
#         return math.log10(self.doc_count / df) if df else 0

#     def calculate_lnc(self, freq):
#         return 1 + math.log10(freq) if freq > 0 else 0

#     def calculate_ltc(self, freq, idf):
#         return (1 + math.log10(freq)) * idf if freq > 0 else 0

#     def calculate_doc_lengths(self):
#         for doc_id in set([doc_id for term in self.dictionary for doc_id, _ in self.dictionary[term]]):
#             length = 0
#             for term, postings in self.dictionary.items():
#                 for posting_doc_id, freq in postings:
#                     if posting_doc_id == doc_id:
#                         lnc = self.calculate_lnc(freq)
#                         length += lnc ** 2
#             self.doc_lengths[doc_id] = math.sqrt(length)
#         print(f"Calculated lengths for {len(self.doc_lengths)} documents")  # Debugging

#     def rank_documents(self, query):
#         query_terms = pre_processing_function(query)
#         print(f"Preprocessed query terms: {query_terms}")

#         query_weights = {}
#         query_length = 0

#         for term in set(query_terms):
#             freq = query_terms.count(term)
#             idf = self.calculate_idf(term)
#             ltc = self.calculate_ltc(freq, idf)
#             query_weights[term] = ltc
#             query_length += ltc ** 2
#             print(f"Query term '{term}': freq={freq}, idf={idf}, ltc={ltc}")  # Debugging

#         query_length = math.sqrt(query_length)
#         print(f"Query length: {query_length}")  # Debugging

#         if query_length == 0:
#             print("Query length is zero after processing.")
#             return []

#         scores = defaultdict(float)
#         for term, q_weight in query_weights.items():
#             if term in self.dictionary:
#                 print(f"Term '{term}' found in {len(self.dictionary[term])} documents")
#                 for doc_id, freq in self.dictionary[term]:
#                     lnc = self.calculate_lnc(freq)
#                     doc_length = self.doc_lengths.get(doc_id, 0)
#                     if doc_length > 0:
#                         score_contribution = (lnc * q_weight) / (doc_length * query_length)
#                         scores[doc_id] += score_contribution
#                         print(f"  Document {doc_id}: lnc={lnc}, doc_length={doc_length}, score_contribution={score_contribution}")
#             else:
#                 print(f"Term '{term}' not found in any document")

#         if not scores:
#             print("No documents matched any query term.")
#             return []

#         print(f"Total documents scored: {len(scores)}")  # Debugging
#         for doc_id, score in list(scores.items())[:5]:  # Print top 5 scores for debugging
#             print(f"Document {doc_id}: final score = {score}")

#         ranked_docs = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
#         return ranked_docs[:10]

# def load_corpus(corpus_dir):
#     vsm = VectorSpaceModel()
#     for filename in os.listdir(corpus_dir):
#         if filename.endswith(".txt"):
#             doc_id = os.path.splitext(filename)[0]
#             filepath = os.path.join(corpus_dir, filename)
#             with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
#                 text = file.read()
#                 vsm.add_document(doc_id, text)
#     vsm.calculate_doc_lengths()
#     print(f"Loaded {vsm.doc_count} documents")
#     print(f"Dictionary contains {len(vsm.dictionary)} unique terms")
#     return vsm

# # Main function
# corpus_dir = r"C:\Users\ravee\Downloads\Corpus"
# vsm = load_corpus(corpus_dir)

# # Input query from user
# query = input("Enter your query: ")

# # Rank documents based on the query
# top_docs = vsm.rank_documents(query)

# # Output top 10 documents by relevance
# if top_docs:
#     for rank, (doc_id, score) in enumerate(top_docs, start=1):
#         print(f"{rank}. ('{doc_id}', {score:.6f})")
# else:
#     print("No matching documents found for the given query.")
import os
import math
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase and tokenize
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Remove stopwords and stem
    return [stemmer.stem(token) for token in tokens if token not in stop_words]

def create_inverted_index(directory):
    inverted_index = defaultdict(lambda: defaultdict(int))
    document_lengths = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                tokens = preprocess_text(content)
                document_lengths[filename] = len(tokens)
                
                for token in set(tokens):
                    inverted_index[token][filename] = tokens.count(token)
    
    return inverted_index, document_lengths

def compute_weights(inverted_index, document_lengths, N):
    weights = defaultdict(lambda: defaultdict(float))
    
    for term, postings in inverted_index.items():
        idf = math.log10(N / len(postings))
        for doc, tf in postings.items():
            tf_weight = 1 + math.log10(tf)
            weights[term][doc] = tf_weight * idf
    
    # Normalize weights
    for doc in document_lengths:
        magnitude = math.sqrt(sum(weights[term][doc]**2 for term in weights if doc in weights[term]))
        for term in weights:
            if doc in weights[term]:
                weights[term][doc] /= magnitude
    
    return weights

def process_query(query, inverted_index, N):
    query_terms = preprocess_text(query)
    query_weights = {}
    
    for term in set(query_terms):
        tf = query_terms.count(term)
        tf_weight = 1 + math.log10(tf)
        idf = math.log10(N / len(inverted_index[term])) if term in inverted_index else 0
        query_weights[term] = tf_weight * idf
    
    # Normalize query weights
    magnitude = math.sqrt(sum(weight**2 for weight in query_weights.values()))
    for term in query_weights:
        query_weights[term] /= magnitude
    
    return query_weights

def cosine_similarity(query_weights, doc_weights):
    return sum(query_weights[term] * doc_weights[term] for term in query_weights if term in doc_weights)

def search(query, inverted_index, weights, N):
    query_weights = process_query(query, inverted_index, N)
    scores = {}
    
    for term in query_weights:
        if term in inverted_index:
            for doc in inverted_index[term]:
                if doc not in scores:
                    scores[doc] = 0
                scores[doc] += query_weights[term] * weights[term][doc]
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

# Main execution
if __name__ == "__main__":
    directory = r"C:\Users\ravee\Downloads\Corpus"
    inverted_index, document_lengths = create_inverted_index(directory)
    N = len(document_lengths)
    weights = compute_weights(inverted_index, document_lengths, N)
    
    query = input("Enter your search query: ")
    results = search(query, inverted_index, weights, N)
    
    print("Top 10 most relevant documents:")
    for doc, score in results:
        print(f"('{doc}', {score})")
