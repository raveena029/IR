import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ... (keep the preprocess function as is)
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
# Modified inverted index creation
'''
def create_inverted_index(folder_path):
    inverted_index = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            tokens = preprocess(content)
            
            for position, token in enumerate(tokens):
                if token in inverted_index:
                    if filename in inverted_index[token]:
                        inverted_index[token][filename].append(position)
                    else:
                        inverted_index[token][filename] = [position]
                else:
                    inverted_index[token] = {filename: [position]}
    
    return inverted_index

# ... (keep other functions as i
def process_proximity_query(query, inverted_index, proximity):
    tokens = preprocess(query)
    tokens = [token for token in tokens if token not in {'and'}]
    
    if len(tokens) != 2:
        print("Error: Proximity query should contain exactly two terms.")
        return {}
    
    token1, token2 = tokens
    
    docs1 = inverted_index.get(token1, {})
    docs2 = inverted_index.get(token2, {})
    
    common_docs = set(docs1.keys()) & set(docs2.keys())
    
    result = {}
    for doc in common_docs:
        positions1 = docs1[doc]
        positions2 = docs2[doc]
        
        for pos1 in positions1:
            for pos2 in positions2:
                if abs(pos1 - pos2) <= proximity:
                    if doc in result:
                        result[doc].append(abs(pos1 - pos2))
                    else:
                        result[doc] = [abs(pos1 - pos2)]
                    break
    
    return result
'''
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
# Modified main function
'''
def main():
    global folder_path
    folder_path = r"C:\Users\ravee\Downloads\Corpus"
    
    inverted_index = create_inverted_index(folder_path)
    total_docs = set(os.listdir(folder_path))
    
    while True:
        query_type = input("Enter query type (boolean/biword/proximity) or 'exit' to quit: ").lower()
        
        if query_type == 'exit':
            break
        
        if query_type == 'boolean':
            query = input("Enter your Boolean query: ")
            result_docs = process_query(query, inverted_index, total_docs)
            if result_docs:
                print("Documents matching the query:", ', '.join(result_docs))
            else:
                print("No documents match the query.")
        elif query_type == 'biword':
            query = input("Enter your Biword query: ")
            result_docs = process_biword_query(query, inverted_index, total_docs)
            if result_docs:
                print("Documents matching the query:", ', '.join(result_docs))
            else:
                print("No documents match the query.")
        elif query_type == 'proximity':
            query = input("Enter your Proximity query: ")
            proximity = int(input("Enter the proximity (number of words): "))
            result = process_proximity_query(query, inverted_index, proximity)
            if result:
                for doc, distances in result.items():
                    print(f"Document: {doc}, Distances: {distances}")
            else:
                print("No documents match the proximity query.")
        else:
            print("Invalid query type. Please enter 'boolean', 'biword', or 'proximity'.")

if __name__ == "__main__":
    main()
'''
def create_inverted_index(folder_path):
    inverted_index = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            tokens = preprocess(content)
            
            for position, token in enumerate(tokens):
                if token in inverted_index:
                    if filename in inverted_index[token]:
                        inverted_index[token][filename].append(position)
                    else:
                        inverted_index[token][filename] = [position]
                else:
                    inverted_index[token] = {filename: [position]}
    
    return inverted_index

def process_proximity_query(query, inverted_index, proximity):
    tokens = preprocess(query)
    tokens = [token for token in tokens if token not in {'and'}]
    
    if len(tokens) != 2:
        print("Error: Proximity query should contain exactly two terms.")
        return {}
    
    token1, token2 = tokens
    
    docs1 = inverted_index.get(token1, {})
    docs2 = inverted_index.get(token2, {})
    
    common_docs = set(docs1.keys()) & set(docs2.keys())
    
    result = {}
    for doc in common_docs:
        positions1 = docs1[doc]
        positions2 = docs2[doc]
        
        for pos1 in positions1:
            for pos2 in positions2:
                distance = abs(pos2 - pos1) - 1  # Subtract 1 to get words between
                if distance <= proximity:
                    if doc in result:
                        result[doc].append(distance)
                    else:
                        result[doc] = [distance]
                    break
    
    return result

def main():
    global folder_path
    folder_path = r"C:\Users\ravee\Downloads\Corpus"
    
    inverted_index = create_inverted_index(folder_path)
    total_docs = set(os.listdir(folder_path))
    
    while True:
        query_type = input("Enter query type (boolean/biword/proximity) or 'exit' to quit: ").lower()
        
        if query_type == 'exit':
            break
        
        if query_type == 'boolean':
            # ... (keep existing boolean query logic)
        elif query_type == 'biword':
            # ... (keep existing biword query logic)
        elif query_type == 'proximity':
            query = input("Enter your Proximity query: ")
            proximity = int(input("Enter the proximity (number of words): "))
            result = process_proximity_query(query, inverted_index, proximity)
            if result:
                for doc, distances in result.items():
                    print(f"Document: {doc}, Words between: {distances}")
            else:
                print("No documents match the proximity query.")
        else:
            print("Invalid query type. Please enter 'boolean', 'biword', or 'proximity'.")

if __name__ == "__main__":
    main()
