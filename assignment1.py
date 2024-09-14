import os
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    tokens = [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word) != '']
    return tokens

def soundex(name):
    name = name.upper()
    soundex = ""
    if name:
        soundex = name[0]
    
    # Remove all occurrences of 'H' and 'W' except first letter
    #name = name[1:].replace('H', '').replace('W', '')
    
    encodings = {
        'AEIOUHWY': '0', 'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
        'L': '4', 'MN': '5', 'R': '6'
    }
    
    for char in name:
        for key in encodings:
            if char in key:
                code = encodings[key]
                if code != soundex[-1]:
                    soundex += code
                break
        if len(soundex) == 4:
            break
    
    soundex = soundex.ljust(4, '0')
    return soundex

def create_indexes(folder_path):
    inverted_index = defaultdict(lambda: defaultdict(list))
    biword_index = defaultdict(lambda: defaultdict(list))
    soundex_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                
            tokens = preprocess(content)
            
            for position, token in enumerate(tokens):
                # Regular inverted index
                inverted_index[token][filename].append(position)
                
                # Biword index
                if position < len(tokens) - 1:
                    biword = f"{token} {tokens[position + 1]}"
                    biword_index[biword][filename].append(position)
                
                # Soundex index
                soundex_code = soundex(token)
                soundex_index[soundex_code][token][filename].append(position)
    
    return inverted_index, biword_index, soundex_index

def boolean_and(list1, list2):
    return list1.intersection(list2)

def boolean_or(list1, list2):
    return list1.union(list2)

def boolean_not(list1, total_docs):
    return total_docs - list1

def process_boolean_query(query, inverted_index, total_docs):
    tokens = query.lower().split()
    result = set()
    current_op = 'and'
    negate_next = False
    first_term = True

    for token in tokens:
        if token in {'and', 'or', 'not'}:
            if token == 'not':
                negate_next = True
            else:
                current_op = token
        else:
            # Preprocess the token
            processed_tokens = preprocess(token)
            if processed_tokens:
                token = processed_tokens[0]  # Take the first processed token
                posting_list = set(inverted_index[token].keys()) if token in inverted_index else set()

                if negate_next:
                    posting_list = total_docs - posting_list
                    negate_next = False

                if first_term:
                    result = posting_list
                    first_term = False
                elif current_op == 'and':
                    result = boolean_and(result, posting_list)
                elif current_op == 'or':
                    result = boolean_or(result, posting_list)

    return result

def process_biword_query(query, biword_index):
    tokens = preprocess(query)
    result_docs = set()
    
    for i in range(len(tokens) - 1):
        biword = f"{tokens[i]} {tokens[i+1]}"
        if biword in biword_index:
            if not result_docs:
                result_docs = set(biword_index[biword].keys())
            else:
                result_docs &= set(biword_index[biword].keys())
    
    return result_docs

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

def process_soundex_query(query, soundex_index):
    tokens = preprocess(query)
    result_docs = set()
    
    for token in tokens:
        soundex_code = soundex(token)
        if soundex_code in soundex_index:
            for word in soundex_index[soundex_code]:
                result_docs.update(soundex_index[soundex_code][word].keys())
    
    return result_docs

def main():
    folder_path = r"C:\Users\ravee\Downloads\Corpus"
    
    inverted_index, biword_index, soundex_index = create_indexes(folder_path)
    total_docs = set(os.listdir(folder_path))
    
    print(f"Indexed {len(total_docs)} documents.")
    print(f"Inverted index contains {len(inverted_index)} unique terms.")
    print(f"Biword index contains {len(biword_index)} unique biwords.")
    print(f"Soundex index contains {len(soundex_index)} unique codes.")
    
    while True:
        query_type = input("Enter query type (boolean/biword/proximity/soundex) or 'exit' to quit: ").lower()
        
        if query_type == 'exit':
            break
        
        if query_type == 'boolean':
            query = input("Enter your Boolean query: ")
            result_docs = process_boolean_query(query, inverted_index, total_docs)
        elif query_type == 'biword':
            query = input("Enter your Biword query: ")
            result_docs = process_biword_query(query, biword_index)
        elif query_type == 'proximity':
            query = input("Enter your Proximity query: ")
            proximity = int(input("Enter the proximity (number of words): "))
            result_docs = process_proximity_query(query, inverted_index, proximity)
            # if result:
            #     for doc, distances in result.items():
            #         print(f"Document: {doc}, Words between: {distances}")
            # else:
            #     print("No documents match the proximity query.")
        elif query_type == 'soundex':
            query = input("Enter your Soundex query: ")
            result_docs = process_soundex_query(query, soundex_index)
        else:
            print("Invalid query type. Please enter 'boolean', 'biword', 'proximity', or 'soundex'.")
            continue
        
        if result_docs:
            print("Documents matching the query:", ', '.join(result_docs))
        else:
            print("No documents match the query.")

if __name__ == "__main__":
    main()
