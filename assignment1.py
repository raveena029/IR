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

# Processing the query
def process_query(query, inverted_index, total_docs):
    tokens = query.lower().split()
    result = None
    i = 0
    
    while i < len(tokens):
        if tokens[i] not in {'and', 'or', 'not'}:
            # Get the posting list for the current term
            term = tokens[i]
            #print(f"Processing term: {term}")  # Debugging line
            posting_list = inverted_index.get(term, set())
            #print(f"Posting list for {term}: {posting_list}")  # Debugging line
            
            if result is None:
                result = posting_list
            else:
                if tokens[i-1] == 'and':
                    result = boolean_and(result, posting_list)
                elif tokens[i-1] == 'or':
                    result = boolean_or(result, posting_list)
        elif tokens[i] == 'not':
            # Handle NOT operator
            i += 1
            term = tokens[i]
            posting_list = inverted_index.get(term, set())
            result = boolean_not(posting_list, total_docs)
        
        i += 1
    
    return result

# Main function
def main():
    # Hardcoded path to the folder with .txt files
    folder_path = r"C:\Users\ravee\Downloads\Corpus"
    
    inverted_index = create_inverted_index(folder_path)
    total_docs = set(os.listdir(folder_path))
    
    # Print the inverted index for debugging purposes
    #print("Inverted Index:", inverted_index)
    
    while True:
        query = input("Enter your Boolean query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        result_docs = process_query(query, inverted_index, total_docs)
        
        if result_docs:
            print("Documents matching the query:", ', '.join(result_docs))
        else:
            print("No documents match the query.")

if __name__ == "__main__":
    main()
