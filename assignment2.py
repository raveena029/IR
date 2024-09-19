import os
import re
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict

def calculate_tfidf(term, doc, inverted_index, doc_lengths, N):
    tf = 1 + math.log10(inverted_index[term][doc])
    idf = math.log10(N / len(inverted_index[term]))
    return tf * idf

def create_inverted_index(folder_path):
    inverted_index = {}
    doc_lengths = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Preprocess the content to get tokens
            tokens = preprocess(content)
            
            for token in tokens:
                if token in inverted_index:
                    if filename in inverted_index[token]:
                        inverted_index[token][filename] += 1
                    else:
                        inverted_index[token][filename] = 1
                else:
                    inverted_index[token] = {filename: 1}
            
            # Calculate document length
            doc_lengths[filename] = len(tokens)
    
    return inverted_index, doc_lengths

def preprocess(text):
    # Case folding (lowercasing)
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming (using PorterStemmer)
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    
    # Normalization (remove punctuation, special chars, etc.)
    tokens = [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word) != '']
    
    return tokens

def cosine_similarity(query_vector, doc_vector):
    dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in set(query_vector) | set(doc_vector))
    query_magnitude = math.sqrt(sum(value ** 2 for value in query_vector.values()))
    doc_magnitude = math.sqrt(sum(value ** 2 for value in doc_vector.values()))
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    
    return dot_product / (query_magnitude * doc_magnitude)

def vsm_search(query, inverted_index, doc_lengths, N):
    query_tokens = preprocess(query)
    query_vector = {}
    for token in query_tokens:
        query_vector[token] = query_vector.get(token, 0) + 1

    doc_scores = defaultdict(lambda: defaultdict(float))
    for term in query_vector:
        if term in inverted_index:
            idf = math.log(N / len(inverted_index[term]))
            for doc in inverted_index[term]:
                tf = inverted_index[term][doc]
                doc_scores[doc][term] = (1 + math.log(tf)) * idf

    results = []
    for doc in doc_scores:
        similarity = cosine_similarity(query_vector, doc_scores[doc])
        results.append((doc, similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)

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
