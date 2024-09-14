import streamlit as st
import os
import re
import base64
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def pre_processing_function(text):
    """
    Preprocess text by lowercasing, removing stopwords, stemming, and removing non-alphanumeric characters.
    
    Args:
    text (str): The input text to be preprocessed
    
    Returns:
    list: A list of preprocessed tokens
    """
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    tokens = [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word) != '']
    return tokens

def create_indexes(folder_path):
    """
    Create inverted index, biphrase index, and soundex index from the given folder of documents.
    
    Args:
    folder_path (str): Path to the folder containing text documents.
    
    Returns:
    tuple: A tuple containing the inverted index, biphrase index, and soundex index of the inputted text documents
    """
    inverted_index = defaultdict(lambda: defaultdict(list))
    biphrase_index = defaultdict(lambda: defaultdict(list))
    soundex_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                
            tokens = pre_processing_function(content)
            
            for position, token in enumerate(tokens):
                # Populating the regular inverted index
                inverted_index[token][filename].append(position)
                
                # Populating the biphrase index for pairs of consecutive tokens in the text
                if position < len(tokens) - 1:
                    biphrase = f"{token} {tokens[position + 1]}"
                    biphrase_index[biphrase][filename].append(position)
                
                # Populating the soundex index
                soundex_code = soundex(token)
                soundex_index[soundex_code][token][filename].append(position)
    
    return inverted_index, biphrase_index, soundex_index

def boolean_and(list1, list2):
    """Perform Boolean AND operation on two sets of documents to find documents having both the query terms"""
    return list1.intersection(list2)

def boolean_or(list1, list2):
    """Perform Boolean OR operation on two sets of documents to find the documents having either of the query terms"""
    return list1.union(list2)

def boolean_not(list1, total_docs):
    """Perform Boolean NOT operation to exclude documents that contains the query term"""
    return total_docs - list1

def process_boolean_query(query, inverted_index, total_docs):
    """
    Process Boolean query with AND, OR, and NOT operators on inverted index.
    
    Args:
    query (str): The boolean query string.
    inverted_index (dict): The inverted index of the document collection.
    total_docs (set): Set of all document IDs in the collection.
    
    Returns:
    set: A set of documents matching the boolean query.
    """
    tokens = query.lower().split()
    matched_docs = set()
    default_operation = 'and'
    not_op = False
    first_term = True

    for token in tokens:
        if token in {'and', 'or', 'not'}:
            if token == 'not':
                not_op = True
            else:
                default_operation = token
        else:
            # Preprocessing the token
            processed_tokens = pre_processing_function(token)
            if processed_tokens:
                token = processed_tokens[0]
                term_postinglist = set(inverted_index[token].keys()) if token in inverted_index else set()

                if not_op:
                    term_postinglist = total_docs - term_postinglist
                    not_op = False

                if first_term:
                    matched_docs = term_postinglist
                    first_term = False
                elif default_operation == 'and':
                    matched_docs = boolean_and(matched_docs, term_postinglist)
                elif default_operation == 'or':
                    matched_docs = boolean_or(matched_docs, term_postinglist)

    return matched_docs

def biphrase_processing_function(query, biphrase_index):
    """
    Process biphrase query by finding documents that contain consecutive phrase pairs.
    
    Args:
    query (str): The biphrase query string.
    biphrase_index (dict): The biphrase index of the document collection.
    
    Returns:
    set: A set of documents matching the biphrase query.
    """
    tokens = pre_processing_function(query)
    matched_docs = set()
    
    for i in range(len(tokens) - 1):
        biphrase = f"{tokens[i]} {tokens[i+1]}"
        if biphrase in biphrase_index:
            if not matched_docs:
                matched_docs = set(biphrase_index[biphrase].keys())
            else:
                matched_docs &= set(biphrase_index[biphrase].keys())
    
    return matched_docs

def soundex(name):
    """
    Implement the Soundex algorithm to convert words into soundex codes for spelling matching.
    
    Args:
    name (str): The input word to be converted to Soundex code.
    
    Returns:
    str: The Soundex code of the input word.
    """
    name = name.upper()
    soundex = name[0]
    
    conversions = {
        'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
        'L': '4', 'MN': '5', 'R': '6'
    }
    
    for char in name[1:]:
        for key in conversions:
            if char in key:
                code = conversions[key]
                if code != soundex[-1]:
                    soundex += code
                break
        if len(soundex) == 4:
            break
    
    return soundex.ljust(4, '0')

def proximity_processing_function(query, inverted_index, proximity):
    """
    Process proximity query to find documents where two terms appear within a certain distance.
    
    Args:
    query (str): The proximity query string input of user
    inverted_index (dict): The inverted index of the collection of document
    proximity (int): The maximum allowed distance between terms as specified by the user
    
    Returns:
    dict: A dictionary of documents and their corresponding word distances.
    """
    tokens = pre_processing_function(query)
    tokens = [token for token in tokens if token not in {'and'}]
    
    if len(tokens) != 2:
        print("Error! Proximity query must have exactly 2 terms to find the proximity")
        return {}
    
    token1, token2 = tokens
    docs1 = inverted_index.get(token1, {})
    docs2 = inverted_index.get(token2, {})
    
    common_docs = set(docs1.keys()) & set(docs2.keys())
    
    matched_docs = {}
    for doc in common_docs:
        positions1 = docs1[doc]
        positions2 = docs2[doc]
        
        for pos1 in positions1:
            for pos2 in positions2:
                distance = abs(pos2 - pos1) - 1
                if distance <= proximity:
                    if doc in matched_docs:
                        matched_docs[doc].append(distance)
                    else:
                        matched_docs[doc] = [distance]
                    break
    
    return matched_docs

def soundex_processing_function(query, soundex_index, inverted_index):
    """
    Process Soundex query for spelling matches and find documents for similar-sounding words.
    
    Args:
    query (string): The soundex query string to find
    soundex_index (dict): The soundex index of the document collection.
    inverted_index (dict): The inverted index of the document collection.
    
    Returns:
    tuple: A tuple containing a set of matching documents and a dictionary of matched words.
    """
    tokens = query.lower().split()
    matched_docs = set()
    matched_words = {}
    
    for token in tokens:
        if token not in {'and', 'or', 'not'}:
            soundex_code = soundex(token)
            similar_words = soundex_index.get(soundex_code, {})
            token_matched_docs = set()
            token_matched_words = set()
            for word in similar_words:
                if word in inverted_index:
                    token_matched_docs.update(inverted_index[word].keys())
                    token_matched_words.add(word)
            if not matched_docs:
                matched_docs = token_matched_docs
                matched_words[token] = token_matched_words
            else:
                matched_docs.intersection_update(token_matched_docs)
                matched_words[token] = token_matched_words
    
    return matched_docs, matched_words

def main():
    """
    Streamlit app: Setting up an interactive interface for users to search queries of their choice
    """
    st.title("Info-Web")

    # Initializing the session state to track index creation by taking input folder directory
    if 'indexes_created' not in st.session_state:
        st.session_state.indexes_created = False

    # Sidebar for folder path input i.e. the corpus to run query on
    folder_path = st.sidebar.text_input("Enter the path to your corpus folder:", r"C:\Users\ravee\Downloads\Corpus")

    # Creating the indexes on button click of the "Create Indexes" button
    if st.sidebar.button("Create Indexes"):
        with st.spinner("Creating indexes..."):
            inverted_index, biphrase_index, soundex_index = create_indexes(folder_path)
            total_docs = set(os.listdir(folder_path))
            st.session_state.inverted_index = inverted_index
            st.session_state.biphrase_index = biphrase_index
            st.session_state.soundex_index = soundex_index
            st.session_state.total_docs = total_docs
            st.session_state.indexes_created = True
        st.success("Indexes created successfully!")

    # Displaying the four options for query types 
    if st.session_state.indexes_created:
        query_type = st.selectbox("Select Query Type", ["Boolean Query", "Biword Query", "Proximity Query", "Soundex Query"])

        # Processing Boolean queries
        if query_type == "Boolean Query":
            query = st.text_input("Enter your Boolean query:")
            if st.button("Search"):
                matched_docs = process_boolean_query(query, st.session_state.inverted_index, st.session_state.total_docs)
                display_matched_docs(matched_docs, folder_path)

        # Processing Biphrase queries
        elif query_type == "Biphrase Query":
            query = st.text_input("Enter your Biword query:")
            if st.button("Search"):
                matched_docs = biphrase_processing_function(query, st.session_state.biphrase_index)
                display_matched_docs(matched_docs, folder_path)

        # Processing Proximity queries
        elif query_type == "Proximity Query":
            query = st.text_input("Enter your Proximity query:")
            proximity = st.number_input("Enter the proximity (in terms of number of words):", min_value=1, value=1)
            if st.button("Search"):
                matched_docs = proximity_processing_function(query, st.session_state.inverted_index, proximity)
                display_matched_docs(matched_docs.keys(), folder_path)

        # Processing Soundex queries
        elif query_type == "Soundex Query":
            query = st.text_input("Enter your Soundex query:")
            if st.button("Search"):
                matched_docs, matched_words = soundex_processing_function(query, st.session_state.soundex_index, st.session_state.inverted_index)
                display_matched_docs(matched_docs, folder_path)

    else:
        st.warning("Please create indexes first by entering the corpus path present in your system and clicking 'Create Indexes' button.")

def display_matched_docs(matched_docs, folder_path):
    """
    Display search results in the app, including document preview and download link.
    
    Args:
    matched_docs (set): Set of documents to display.
    folder_path (string): Path to the folder containing the documents of the corpus
    """
    if matched_docs:
        st.write(f"{len(matched_docs)} documents matching the query found:")
        for doc in matched_docs:
            filepath = os.path.join(folder_path, doc)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                doc_preview = file.read(500) + "..."
                st.write(f"Document: {doc}")
                st.text_area(f"Preview of {doc}", doc_preview, height=100)
                
                # Download button for each document
                with open(filepath, "rb") as file:
                    doc_bytes = file.read()
                    b64 = base64.b64encode(doc_bytes).decode()
                    doc_location_path = f'<a href="data:file/txt;base64,{b64}" download="{doc}">Download {doc}</a>'
                    st.markdown(doc_location_path, unsafe_allow_html=True)
    else:
        st.write("No documents matched with the specified query.")

# Run the Streamlit app
if __name__ == "__main__":
    main()





# import os
# import re
# from collections import defaultdict
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

# def preprocess(text):
#     text = text.lower()
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
#     ps = PorterStemmer()
#     tokens = [ps.stem(word) for word in tokens]
#     tokens = [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word) != '']
#     return tokens


# def create_indexes(folder_path):
#     inverted_index = defaultdict(lambda: defaultdict(list))
#     biword_index = defaultdict(lambda: defaultdict(list))
#     soundex_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             filepath = os.path.join(folder_path, filename)
            
#             with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
#                 content = file.read()
                
#             tokens = preprocess(content)
            
#             for position, token in enumerate(tokens):
#                 inverted_index[token][filename].append(position)
                
#                 if position < len(tokens) - 1:
#                     biword = f"{token} {tokens[position + 1]}"
#                     biword_index[biword][filename].append(position)
                
#                 soundex_code = soundex(token)
#                 soundex_index[soundex_code][token][filename].append(position)
    
#     return inverted_index, biword_index, soundex_index

# def boolean_and(list1, list2):
#     return list1.intersection(list2)

# def boolean_or(list1, list2):
#     return list1.union(list2)

# def boolean_not(list1, total_docs):
#     return total_docs - list1

# def process_boolean_query(query, inverted_index, total_docs):
#     tokens = query.lower().split()
#     result = set()
#     current_op = 'and'
#     negate_next = False
#     first_term = True

#     for token in tokens:
#         if token in {'and', 'or', 'not'}:
#             if token == 'not':
#                 negate_next = True
#             else:
#                 current_op = token
#         else:
#             processed_tokens = preprocess(token)
#             if processed_tokens:
#                 token = processed_tokens[0]  
#                 posting_list = set(inverted_index[token].keys()) if token in inverted_index else set()

#                 if negate_next:
#                     posting_list = total_docs - posting_list
#                     negate_next = False

#                 if first_term:
#                     result = posting_list
#                     first_term = False
#                 elif current_op == 'and':
#                     result = boolean_and(result, posting_list)
#                 elif current_op == 'or':
#                     result = boolean_or(result, posting_list)

#     return result

# def process_biword_query(query, biword_index):
#     tokens = preprocess(query)
#     result_docs = set()
    
#     for i in range(len(tokens) - 1):
#         biword = f"{tokens[i]} {tokens[i+1]}"
#         if biword in biword_index:
#             if not result_docs:
#                 result_docs = set(biword_index[biword].keys())
#             else:
#                 result_docs &= set(biword_index[biword].keys())
    
#     return result_docs

# def soundex(name):
#     name = name.upper()
#     soundex = name[0]
    
#     conversions = {
#         'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
#         'L': '4', 'MN': '5', 'R': '6'
#     }
    
#     for char in name[1:]:
#         for key in conversions:
#             if char in key:
#                 code = conversions[key]
#                 if code != soundex[-1]:  
#                     soundex += code
#                 break
#         if len(soundex) == 4:
#             break
    
#     soundex = soundex.ljust(4, '0')
    
#     return soundex

# def process_proximity_query(query, inverted_index, proximity):
#     tokens = preprocess(query)
#     tokens = [token for token in tokens if token not in {'and'}]
    
#     if len(tokens) != 2:
#         print("Error: Proximity query should contain exactly two terms.")
#         return {}
    
#     token1, token2 = tokens
    
#     docs1 = inverted_index.get(token1, {})
#     docs2 = inverted_index.get(token2, {})
    
#     common_docs = set(docs1.keys()) & set(docs2.keys())
    
#     result = {}
#     for doc in common_docs:
#         positions1 = docs1[doc]
#         positions2 = docs2[doc]
        
#         for pos1 in positions1:
#             for pos2 in positions2:
#                 distance = abs(pos2 - pos1) - 1  
#                 if distance <= proximity:
#                     if doc in result:
#                         result[doc].append(distance)
#                     else:
#                         result[doc] = [distance]
#                     break
    
#     return result

# def process_soundex_query(query, soundex_index, inverted_index):
#     tokens = query.lower().split()
#     result = set()
#     matched_words = {}
    
#     for token in tokens:
#         if token not in {'and', 'or', 'not'}:
#             soundex_code = soundex(token)
#             similar_words = soundex_index.get(soundex_code, set())
#             token_result = set()
#             token_matched_words = set()
#             for word in similar_words:
#                 if word in inverted_index:
#                     token_result.update(inverted_index[word])
#                     token_matched_words.add(word)
#             if not result:
#                 result = token_result
#                 matched_words[token] = token_matched_words
#             else:
#                 result.intersection_update(token_result)
#                 matched_words[token] = token_matched_words
    
#     return result, matched_words

# def create_soundex_index(inverted_index):
#     soundex_index = {}
#     for word in inverted_index:
#         soundex_code = soundex(word)
#         if soundex_code in soundex_index:
#             soundex_index[soundex_code].add(word)
#         else:
#             soundex_index[soundex_code] = {word}
#     return soundex_index

# def main():
#     folder_path = r"C:\Users\ravee\Downloads\Corpus"
    
#     inverted_index, biword_index, soundex_index = create_indexes(folder_path)
#     total_docs = set(os.listdir(folder_path))
    
#     print(f"Indexed {len(total_docs)} documents.")
#     print(f"Inverted index contains {len(inverted_index)} unique terms.")
#     print(f"Biword index contains {len(biword_index)} unique biwords.")
#     print(f"Soundex index contains {len(soundex_index)} unique codes.")
    
#     while True:
#         query_type = int(input("Enter query type (1-boolean/2-biword/3-proximity/4-soundex) or '5' to quit: "))
        
#         if query_type == 5:
#             break
        
#         if query_type == 1:
#             query = input("Enter your Boolean query: ")
#             result_docs = process_boolean_query(query, inverted_index, total_docs)
#         elif query_type == 2:
#             query = input("Enter your Biword query: ")
#             result_docs = process_biword_query(query, biword_index)
#         elif query_type == 3:
#             query = input("Enter your Proximity query: ")
#             proximity = int(input("Enter the proximity (number of words): "))
#             result_docs = process_proximity_query(query, inverted_index, proximity)
#         elif query_type == 4:
#             query = input("Enter your Soundex query: ")
#             result_docs, matched_words = process_soundex_query(query, soundex_index, inverted_index)
#             # if result_docs:
#             #     print("Matched words:")
#             #     for token, words in matched_words.items():
#             #         print(f"  '{token}' matched with: {', '.join(words)}")
#         else:
#             print("Invalid query type. Please enter 'boolean', 'biword', 'proximity', or 'soundex'.")
#             continue
        
#         if result_docs:
#             print("Documents matching the query:", ', '.join(result_docs))
#         else:
#             print("No documents match the query.")

# if __name__ == "__main__":
#     main()
