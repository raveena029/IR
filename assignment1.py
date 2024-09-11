import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Preprocessing functions
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

# Creating the inverted index
def create_inverted_index(folder_path):
    inverted_index = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            tokens = preprocess(content)
            
            for token in tokens:
                if token in inverted_index:
                    inverted_index[token].add(filename)
                else:
                    inverted_index[token] = {filename}
    
    return inverted_index

# Boolean operations
def boolean_and(list1, list2):
    return list1.intersection(list2)

def boolean_or(list1, list2):
    return list1.union(list2)

def boolean_not(list1, total_docs):
    return total_docs - list1
def process_query(query, inverted_index, total_docs):
    tokens = query.lower().split()
    result = set(total_docs)  # Start with all documents
    current_op = 'and'  # Default operation
    negate_next = False

    for token in tokens:
        if token in {'and', 'or', 'not'}:
            if token == 'not':
                negate_next = True
            else:
                current_op = token
        else:
            posting_list = inverted_index.get(token, set())
            if negate_next:
                posting_list = total_docs - posting_list
                negate_next = False

            if current_op == 'and':
                result = result.intersection(posting_list)
            elif current_op == 'or':
                result = result.union(posting_list)

    return result
# ... (previous code remains the same)

def process_biword_query(query, inverted_index, total_docs):
    # Preprocess the biword query
    tokens = preprocess(query)
    
    if len(tokens) < 2:
        print("Error: Biword query should contain at least two words.")
        return set()
    
    # Get the first two tokens for the biword query
    token1, token2 = tokens[:2]
    
    # Get the posting lists for both tokens
    posting_list1 = inverted_index.get(token1, set())
    posting_list2 = inverted_index.get(token2, set())
    
    # Find documents that contain both tokens
    common_docs = posting_list1.intersection(posting_list2)
    
    # Check for consecutive occurrence in the common documents
    result_docs = set()
    for doc in common_docs:
        filepath = os.path.join(folder_path, doc)
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Preprocess the document content
        doc_tokens = preprocess(content)
        
        # Check for consecutive occurrence
        for i in range(len(doc_tokens) - 1):
            if doc_tokens[i] == token1 and doc_tokens[i+1] == token2:
                result_docs.add(doc)
                break
    
    return result_docs

# Main function
def main():
    # Hardcoded path to the folder with .txt files
    global folder_path
    folder_path = r"C:\Users\ravee\Downloads\Corpus"
    
    inverted_index = create_inverted_index(folder_path)
    total_docs = set(os.listdir(folder_path))
    
    while True:
        query_type = input("Enter query type (boolean/biword) or 'exit' to quit: ").lower()
        
        if query_type == 'exit':
            break
        
        if query_type == 'boolean':
            query = input("Enter your Boolean query: ")
            result_docs = process_query(query, inverted_index, total_docs)
        elif query_type == 'biword':
            query = input("Enter your Biword query: ")
            result_docs = process_biword_query(query, inverted_index, total_docs)
        else:
            print("Invalid query type. Please enter 'boolean' or 'biword'.")
            continue
        
        if result_docs:
            print("Documents matching the query:", ', '.join(result_docs))
        else:
            print("No documents match the query.")

if __name__ == "__main__":
    main()

# def main():
#     folder_path = r"C:\Users\ravee\Downloads\Corpus"
    
#     inverted_index = create_inverted_index(folder_path)
#     total_docs = set(os.listdir(folder_path))
    
#     while True:
#         query = input("Enter your Boolean query (or type 'exit' to quit): ")
#         if query.lower() == 'exit':
#             break
        
#         # Preprocess query terms, but keep 'and', 'or', 'not' as is
#         processed_query = []
#         for token in query.lower().split():
#             if token in {'and', 'or', 'not'}:
#                 processed_query.append(token)
#             else:
#                 processed_query.extend(preprocess(token))
        
#         result_docs = process_query(' '.join(processed_query), inverted_index, total_docs)
        
#         if result_docs:
#             print("Documents matching the query:", ', '.join(result_docs))
#         else:
#             print("No documents match the query.")

# if __name__ == "__main__":
#     main()
