import streamlit as st
import os
import re
import base64
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Include all your existing functions here (preprocess, create_indexes, boolean_and, boolean_or, boolean_not, 
# process_boolean_query, process_biword_query, soundex, process_proximity_query, process_soundex_query, create_soundex_index)

# ... (paste all the functions from your original code here)
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    tokens = [re.sub(r'\W+', '', word) for word in tokens if re.sub(r'\W+', '', word) != '']
    return tokens


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

def soundex(name):
    name = name.upper()
    soundex = name[0]
    
    # Conversion table
    conversions = {
        'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
        'L': '4', 'MN': '5', 'R': '6'
    }
    
    # Convert name to Soundex code
    for char in name[1:]:
        for key in conversions:
            if char in key:
                code = conversions[key]
                if code != soundex[-1]:  # Only add if not the same as the last code
                    soundex += code
                break
        if len(soundex) == 4:
            break
    
    # Pad with zeros if necessary
    soundex = soundex.ljust(4, '0')
    
    return soundex

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

# def process_soundex_query(query, soundex_index, inverted_index):
#     tokens = preprocess(query)
#     tokens = [token for token in tokens if token not in {'and', 'or', 'not'}]
    
#     if len(tokens) != 2:
#         print("Error: Soundex query should contain exactly two terms.")
#         return set()
    
#     token1, token2 = tokens
#     soundex_code1 = soundex(token1)
#     soundex_code2 = soundex(token2)
    
#     similar_words1 = soundex_index.get(soundex_code1, set())
#     similar_words2 = soundex_index.get(soundex_code2, set())
    
#     result_docs = set()
    
#     for word1 in similar_words1:
#         for word2 in similar_words2:
#             if word1 in inverted_index and word2 in inverted_index:
#                 docs1 = set(inverted_index[word1].keys())
#                 docs2 = set(inverted_index[word2].keys())
#                 common_docs = docs1.intersection(docs2)
#                 result_docs.update(common_docs)
    
#     return result_docs
def process_soundex_query(query, soundex_index, inverted_index):
    tokens = query.lower().split()
    result = set()
    matched_words = {}
    
    for token in tokens:
        if token not in {'and', 'or', 'not'}:
            soundex_code = soundex(token)
            similar_words = soundex_index.get(soundex_code, set())
            token_result = set()
            token_matched_words = set()
            for word in similar_words:
                if word in inverted_index:
                    token_result.update(inverted_index[word])
                    token_matched_words.add(word)
            if not result:
                result = token_result
                matched_words[token] = token_matched_words
            else:
                result.intersection_update(token_result)
                matched_words[token] = token_matched_words
    
    return result, matched_words

def create_soundex_index(inverted_index):
    soundex_index = {}
    for word in inverted_index:
        soundex_code = soundex(word)
        if soundex_code in soundex_index:
            soundex_index[soundex_code].add(word)
        else:
            soundex_index[soundex_code] = {word}
    return soundex_index
# Streamlit app
def main():
    st.title("Info-Web")

    # Initialize session state
    if 'indexes_created' not in st.session_state:
        st.session_state.indexes_created = False

    # Sidebar for folder path input
    folder_path = st.sidebar.text_input("Enter the path to your corpus folder:", r"C:\Users\ravee\Downloads\Corpus")

    if st.sidebar.button("Create Indexes"):
        with st.spinner("Creating indexes..."):
            inverted_index, biword_index, soundex_index = create_indexes(folder_path)
            total_docs = set(os.listdir(folder_path))
            st.session_state.inverted_index = inverted_index
            st.session_state.biword_index = biword_index
            st.session_state.soundex_index = soundex_index
            st.session_state.total_docs = total_docs
            st.session_state.indexes_created = True
        st.success("Indexes created successfully!")

    if st.session_state.indexes_created:
        query_type = st.selectbox("Select Query Type", ["Boolean Query", "Biword Query", "Proximity Query", "Soundex Query"])

        if query_type == "Boolean Query":
            query = st.text_input("Enter your Boolean query:")
            if st.button("Search"):
                result_docs = process_boolean_query(query, st.session_state.inverted_index, st.session_state.total_docs)
                display_results(result_docs, folder_path)

        elif query_type == "Biword Query":
            query = st.text_input("Enter your Biword query:")
            if st.button("Search"):
                result_docs = process_biword_query(query, st.session_state.biword_index)
                display_results(result_docs, folder_path)

        elif query_type == "Proximity Query":
            query = st.text_input("Enter your Proximity query:")
            proximity = st.number_input("Enter the proximity (number of words):", min_value=1, value=1)
            if st.button("Search"):
                result_docs = process_proximity_query(query, st.session_state.inverted_index, proximity)
                display_results(result_docs.keys(), folder_path)

        elif query_type == "Soundex Query":
            query = st.text_input("Enter your Soundex query:")
            if st.button("Search"):
                result_docs, matched_words = process_soundex_query(query, st.session_state.soundex_index, st.session_state.inverted_index)
                display_results(result_docs, folder_path)
                # st.write("Matched words:")
                # for token, words in matched_words.items():
                #     st.write(f"  '{token}' matched with: {', '.join(words)}")

    else:
        st.warning("Please create indexes first by entering the corpus folder path and clicking 'Create Indexes'.")

def display_results(result_docs, folder_path):
    if result_docs:
        st.write("Documents matching the query:")
        for doc in result_docs:
            doc_path = os.path.join(folder_path, doc)
            
            # Read the file content
            with open(doc_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            # Create a download link for the file
            b64 = base64.b64encode(content.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{doc}">Download {doc}</a>'
            
            # Display the link and a preview of the content
            st.markdown(href, unsafe_allow_html=True)
            st.text_area(f"Preview of {doc}", content[:500] + "...", height=150)
    else:
        st.write("No documents match the query.")
        
if __name__ == "__main__":
    main()
