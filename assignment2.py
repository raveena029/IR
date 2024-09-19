import os
import re
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    tokens = [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word) != '']
    return tokens

def create_inverted_index(folder_path):
    inverted_index = defaultdict(list)
    doc_lengths = {}
    doc_term_freqs = defaultdict(lambda: defaultdict(int))
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            tokens = preprocess(content)
            doc_lengths[filename] = len(tokens)
            for token in tokens:
                doc_term_freqs[filename][token] += 1
    
    for doc, term_freqs in doc_term_freqs.items():
        for term, freq in term_freqs.items():
            inverted_index[term].append((doc, freq))
    
    return inverted_index, doc_lengths

def calculate_tfidf(term, doc, inverted_index, N, is_query=False):
    tf = 1 + math.log10(next(freq for d, freq in inverted_index[term] if d == doc))
    if is_query:
        idf = math.log10(N / len(inverted_index[term]))
    else:
        idf = 1  # For documents, idf is 1 in lnc scheme
    return tf * idf

def cosine_similarity(query_vector, doc_vector):
    dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in set(query_vector) | set(doc_vector))
    query_magnitude = math.sqrt(sum(value ** 2 for value in query_vector.values()))
    doc_magnitude = math.sqrt(sum(value ** 2 for value in doc_vector.values()))
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    return dot_product / (query_magnitude * doc_magnitude)

def vsm_search(query, inverted_index, doc_lengths, N):
    query_tokens = preprocess(query)
    query_vector = defaultdict(float)
    for token in query_tokens:
        query_vector[token] += 1
    for token in query_vector:
        query_vector[token] = (1 + math.log10(query_vector[token])) * math.log10(N / len(inverted_index[token]))

    doc_vectors = defaultdict(lambda: defaultdict(float))
    for term in query_vector:
        if term in inverted_index:
            for doc, freq in inverted_index[term]:
                doc_vectors[doc][term] = 1 + math.log10(freq)

    # Normalize document vectors
    for doc in doc_vectors:
        magnitude = math.sqrt(sum(value ** 2 for value in doc_vectors[doc].values()))
        for term in doc_vectors[doc]:
            doc_vectors[doc][term] /= magnitude

    results = []
    for doc in doc_vectors:
        similarity = cosine_similarity(query_vector, doc_vectors[doc])
        results.append((doc, similarity))

    return sorted(results, key=lambda x: (-x[1], x[0]))[:10]

def main():
    folder_path = r"C:\Users\ravee\Downloads\Corpus"
    inverted_index, doc_lengths = create_inverted_index(folder_path)
    N = len(doc_lengths)

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        results = vsm_search(query, inverted_index, doc_lengths, N)
        query_tokens = preprocess(query)
        print("Preprocessed query tokens:", query_tokens)
        if results:
            print("Top 10 documents matching the query:")
            for doc, score in results:
                print(f"{doc}: {score}")
        else:
            print("No documents match the query.")

if __name__ == "__main__":
    main()
