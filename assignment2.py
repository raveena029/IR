import os
import math
import re
import streamlit as st
import base64
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Preprocessing function
def func_to_preprocess_text(text):
    """
    Preprocesses the given text by performing the following steps:
        Converts text to lowercase.
        Tokenizes the text into individual words.
        Removes English stopwords.
        Stems each word using the Porter Stemming algorithm.
        Removes any non-alphanumeric characters.
    
    Args:
        text (str): The input text to be preprocessed.
    
    Returns:
        list: A list of preprocessed, stemmed tokken.
    """
    #case folding the tokken
    text = text.lower()
    tokken = word_tokenize(text)
    #removing the stop words
    stop_words = set(stopwords.words('english'))
    tokken = [word for word in tokken if word not in stop_words]
    #perfomring stemming
    ps = PorterStemmer()
    tokken = [ps.stem(word) for word in tokken]
    tokken = [re.sub(r'\W+', '', word) for word in tokken if re.sub(r'\W+', '', word) != '']
    return tokken

class VectorSpaceModel:
    """
    A Vector Space Model (VSM) class for indexing documents and ranking them based on cosine similarity to a query.
    
    Attributes:
        dictionary (defaultdict): A dictionary storing terms and the list of (doc_id, frequency) tuples for each term.
        document_lenggth (defaultdict): Stores the document lengths for normalization.
        document_frequencyy (int): A counter for the number of documents added to the VSM.
    """
    def __init__(self):
        """
        Initializes the Vector Space Model by setting up necessary dictionaries and counters.
        """
        self.dictionary = defaultdict(list)
        self.document_lenggth = defaultdict(float)
        self.document_frequencyy = 0

    def update_docs_in_vsm(self, doc_id, text):
        """
        Adds a document to the VSM by processing its text and updating the term frequencies.
        
        Args:
            doc_id (str): The identifier for the document.
            text (str): The content of the document to be processed.
        """
        tokken = func_to_preprocess_text(text)
        term_freq = Counter(tokken)
        #calculating the term frequency of each term in the document of the corpus 
        for term, freq in term_freq.items():
            self.dictionary[term].append((doc_id, freq))
        self.document_lenggth[doc_id] = math.sqrt(sum((1 + math.log10(freq))**2 for freq in term_freq.values()))
        self.document_frequencyy += 1

    def calculate_inverse_doc_freq(self, term):
        """
        Calculates the Inverse Document Frequency (IDF) for a given term.
        
        Args:
            term (str): The term for which IDF is calculated.
        
        Returns:
            float: The IDF value for the term.
        """
        #calculating the inverse document frequcney to measure the rareness of the document
        return math.log10(self.document_frequencyy / len(self.dictionary[term])) if term in self.dictionary else 0

    def lnc_doc_calculation(self, term, freq):
        """
        Calculates the LNC (Logarithmic term frequency normalization) weight for a term in a document.
        
        Args:
            term (str): The term for which the weight is calculated.
            freq (int): The term frequency in the document.
        
        Returns:
            float: The LNC weight for the term.
        """
        #calculating the term term frequency * 1(since document frequecny is none i.e. 1)
        return 1 + math.log10(freq)

    def ltc_query_calculation(self, term, freq, idf):
        """
        Calculates the LTC (Logarithmic term frequency + IDF normalization) weight for a query term.
        
        Args:
            term (str): The term in the query.
            freq (int): The frequency of the term in the query.
            idf (float): The Inverse Document Frequency (IDF) of the term.
        
        Returns:
            float: The LTC weight for the term.
        """
        # calcualting the tf*idf weight
        return (1 + math.log10(freq)) * idf

    def func_to_rank_documents(self, query):
        """
        Ranks documents based on cosine similarity to the query using term weights.
        
        Args:
            query (str): The search query entered by the user.
        
        Returns:
            list: A list of the top 10 ranked documents (doc_id, score).
            dict: A dictionary of term matches for each document.
        """
        query_terms = func_to_preprocess_text(query)
        query_frequency = Counter(query_terms)
        #calling the functions to calcualte the weighted scores of the query
        weighated_query = {term: self.ltc_query_calculation(term, freq, self.calculate_inverse_doc_freq(term)) for term, freq in query_frequency.items()}
        #normalising the qeighting scores
        query_length_norm = math.sqrt(sum(weight**2 for weight in weighated_query.values()))

        scores = defaultdict(float)
        matched_terms = defaultdict(lambda: defaultdict(list))
        #calculating the cosine simialirty between the document and the query
        for term, query_weight in weighated_query.items():
            if term in self.dictionary:
                for doc_id, freq in self.dictionary[term]:
                    docuemtn_weight = self.lnc_doc_calculation(term, freq)
                    scores[doc_id] += docuemtn_weight * query_weight
                    matched_terms[doc_id][term] = freq  # Changed to freq instead of self.dictionary[term][doc_id]
        #calcualting the normalised scores           
        for doc_id in scores:
            scores[doc_id] /= (self.document_lenggth[doc_id] * query_length_norm)
        #ranking the documents on the basis of the score generated
        docu_ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        #returning the top 10 highest scored document i.e. most relevenat dcument
        return docu_ranking[:10], matched_terms


    # def get_matching_preview(self, doc_id, query, matched_terms, window_size=200):
    #     """
    #     Retrieves a preview of the document showing where query terms match, with highlights.
        
    #     Args:
    #         doc_id (str): The document identifier.
    #         query (str): The search query.
    #         matched_terms (dict): Term matches for the document.
    #         window_size (int): The size of the preview window (default 200 characters).
        
    #     Returns:
    #         tuple: The highlighted preview text, best score, and starting position.
    #     """
    #     # Reading the document from the corpus
    #     with open(os.path.join(self.corpus_dir, doc_id), 'r', encoding='utf-8', errors='ignore') as file:
    #         content = file.read()

    #     # Preprocessing the query into individual terms
    #     query_terms = set(func_to_preprocess_text(query))
        
    #     best_window = None
    #     best_score = 0
    #     best_start = 0

    #     # Create a regex pattern to find the exact query
    #     exact_query_pattern = re.escape(query)
    #     query_pattern = re.compile('|'.join(map(re.escape, query_terms)), re.IGNORECASE)
        
    #     # Search for the exact match of the query in the content
    #     exact_match = re.search(exact_query_pattern, content, re.IGNORECASE)

    #     if exact_match:
    #         best_start = exact_match.start()  # Start position of the exact match
    #         best_end = best_start + len(query)  # End position based on the exact query length

    #         # Adjust the end position to include the desired window size
    #         best_end = min(best_end + (window_size - len(query)), len(content))  # Ensure it doesn't exceed content length

    #         # Extract the window text from the content
    #         best_window = content[best_start:best_end]

    #         # Count the matches in this window
    #         match_count = sum(1 for term in query_terms if term.lower() in best_window.lower())
    #         best_score = match_count / len(query_terms) if query_terms else 0

    #         # Highlight the window
    #         highlighted_window = best_window
    #         highlighted_window = query_pattern.sub(
    #             lambda m: f"<span style='background-color: yellow; text-decoration: underline; color: black;'>{m.group()}</span>", 
    #             highlighted_window
    #         )

    #         return highlighted_window, best_score, best_start
    #     else:
    #         # If no exact match was found, search for the first occurrence of any query term
    #         first_match = query_pattern.search(content)

    #         if first_match:
    #             best_start = first_match.start()
    #             best_end = best_start + window_size  # Default to the window size from the first match

    #             # Ensure we do not exceed the content length
    #             if best_end > len(content):
    #                 best_end = len(content)

    #             # Extract the window text from the content
    #             best_window = content[best_start:best_end]

    #             # Count the matches in this window
    #             match_count = sum(1 for term in query_terms if term.lower() in best_window.lower())
    #             best_score = match_count / len(query_terms) if query_terms else 0

    #             # Highlight the window
    #             highlighted_window = best_window
    #             highlighted_window = query_pattern.sub(
    #                 lambda m: f"<span style='background-color: yellow; text-decoration: underline; color: black;'>{m.group()}</span>", 
    #                 highlighted_window
    #             )

    #             return highlighted_window, best_score, best_start

    #     # If no match was found, return None
    #     return None, best_score, 0


    def get_matching_preview(self, doc_id, query, matched_terms, window_size=200):
        """
        Retrieves a preview of the document showing where query terms match, with highlights.
        
        Args:
            doc_id (str): The document identifier.
            query (str): The search query.
            matched_terms (dict): Term matches for the document.
            window_size (int): The size of the preview window (default 200 characters).
        
        Returns:
            tuple: The highlighted preview text, best score, and starting position.
        """
        # Reading the document from the corpus
        with open(os.path.join(self.corpus_dir, doc_id), 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

        # Preprocessing the query into individual terms
        query_terms = set(func_to_preprocess_text(query))
        
        best_window = None
        best_score = 0
        best_start = 0
        
        # Tokenizing the content of the document while keeping track of original positions
        tokken_with_positions = []
        current_pos = 0
        for token in word_tokenize(content.lower()):
            start = content.lower().find(token, current_pos)
            end = start + len(token)
            tokken_with_positions.append((token, start, end))
            current_pos = end
        
        # Find the most relevant portion of the document matching the query terms
        for i in range(len(tokken_with_positions) - window_size + 1):
            window = tokken_with_positions[i:i + window_size]
            window_text = content[window[0][1]:window[-1][2]]
            match_count = sum(1 for term in query_terms if term in [token for token, _, _ in window])
            score = match_count / len(query_terms)
            
            # Track the best scoring window
            if score > best_score:
                best_score = score
                best_window = window_text
                best_start = window[0][1]
        
        # If a relevant portion is found, highlight the best window
        if best_score > 0:
            # Try to highlight and underline the entire query first
            highlighted_window = best_window
            query_pattern = re.compile(re.escape(query), re.IGNORECASE)
            
            if query_pattern.search(best_window):
                # If the exact query is found, highlight and underline the full query
                highlighted_window = query_pattern.sub(lambda m: f"<span style='background-color: yellow; text-decoration: underline; color: black;'>{m.group()}</span>", highlighted_window)
            else:
                # If the exact query is not found, highlight individual matching query terms
                for term in query_terms:
                    term_pattern = re.compile(re.escape(term), re.IGNORECASE)
                    highlighted_window = term_pattern.sub(lambda m: f"<span style='background-color: yellow; color: black;'>{m.group()}</span>", highlighted_window)

            return highlighted_window, best_score, best_start
        else:
            # If no match was found, return None
            return None, best_score, 0



def func_to_load_corpus_data(corpus_dir):
    """
    Loads the document corpus and adds each document to the Vector Space Model.
    
    Args:
        corpus_dir (str): The directory path containing the documents.
    
    Returns:
        VectorSpaceModel: The initialized Vector Space Model with added documents.
    """
    vsm = VectorSpaceModel()
    vsm.corpus_dir = corpus_dir  # Added this line to define corpus_dir
    #reading the content of the files and sending it to vsm fucntion to calcualte term frequency and posting lists
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            document_path = os.path.join(corpus_dir, filename)
            with open(document_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                vsm.update_docs_in_vsm(filename, text)
    return vsm


def func_to_print_relevant_docs(relevant_documents, corpus_pathh, query, vsm):
    """
    Displays the results of the search query.
        
        Parameters:
        - relevant_documents: List of tuples containing document IDs and their relevance scores.
        - corpus_pathh: Path to the corpus folder where documents are stored.
        - query: The search query entered by the user.
        - vsm: The Vector Space Model object that contains the term matches and document rankings.
    """
    #chceking if relevant docuemnts found in the corupus 
    if relevant_documents:
        st.write("Documents matching the query:")

        # Get the top-ranked document (most relevant)
        top_doc_id, top_score = relevant_documents[0]

        for i, (doc_id, score) in enumerate(relevant_documents):
            doc_path = os.path.join(corpus_pathh, doc_id)

            # Read the file content
            with open(doc_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Create a download link for the file
            b64 = base64.b64encode(content.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{doc_id}">Download {doc_id}</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.write(f"Score: {score:.4f}")

            if i == 0:
                # For the most relevant document, display a matching preview highlightd with color
                preview, match_score, start_pos = vsm.get_matching_preview(top_doc_id, query, vsm.matched_terms[top_doc_id])
                #dispalying a preview of the docuemnt
                if preview:
                    st.markdown(f"**Matching preview of {doc_id}:** {preview}", unsafe_allow_html=True)
                    #st.write(f"Preview starts at character position: {start_pos}")
                else:
                    st.text_area(f"Preview of {doc_id}", content[:500] + "...", height=100)
            else:
                # For other documents, display content without highlighting
                st.text_area(f"Preview of {doc_id}", content[:500] + "...", height=100)          
    else:
        st.write("No documents match the query.")



def main():
    """
        Main function to manage the Streamlit user interface for the Vector Space Model (VSM) search engine.
        Allows users to create the VSM from a folder of documents and perform searches.
    """
    st.title("Info- Web")

    # Initialize session state
    if 'vsm_created' not in st.session_state:
        st.session_state.vsm_created = False

    # Sidebar to enter the folder/corpus path as input
    corpus_pathh = st.sidebar.text_input("Enter the path to your corpus folder:", r"C:\Users\ravee\Downloads\Corpus")

    if st.sidebar.button("Create VSM"):
        with st.spinner("Creating Vector Space Model..."):
            vsm = func_to_load_corpus_data(corpus_pathh)
            st.session_state.vsm = vsm
            st.session_state.vsm_created = True
        st.success("Vector Space Model created successfully!")
    # take the input query from the user and send it for precrossing and cosine cimiarity score calculation
    if st.session_state.vsm_created:
        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            relevant_documents, matched_terms = st.session_state.vsm.func_to_rank_documents(query)
            st.session_state.vsm.matched_terms = matched_terms
            func_to_print_relevant_docs(relevant_documents, corpus_pathh, query, st.session_state.vsm)
    #dispaly warning if there is no path of corpus
    else:
        st.warning("Please create the Vector Space Model first by entering the corpus path from your local device and clicking the 'Create VSM' button.")

if __name__ == "__main__":
    main()



# import os
# import math
# import re
# from collections import defaultdict, Counter
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

# def func_to_preprocess_text(text):
#     text = text.lower()
#     tokken = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     tokken = [word for word in tokken if word not in stop_words]
#     ps = PorterStemmer()
#     tokken = [ps.stem(word) for word in tokken]
#     tokken = [re.sub(r'\W+', '', word) for word in tokken if re.sub(r'\W+', '', word) != '']
#     return tokken if tokken else ['placeholder']

# class VectorSpaceModel:
#     def __init__(self):
#         self.dictionary = defaultdict(list)  # term -> [(doc_id, tf)]
#         self.document_lenggth = defaultdict(float)  # doc_id -> length
#         self.document_frequencyy = 0

#     def update_docs_in_vsm(self, doc_id, text):
#         tokken = func_to_preprocess_text(text)
#         term_freq = Counter(tokken)
#         for term, freq in term_freq.items():
#             self.dictionary[term].append((doc_id, freq))
#         self.document_frequencyy += 1

#     def calculate_inverse_doc_freq(self, term):
#         df = len(self.dictionary[term])
#         return math.log10(self.document_frequencyy / df) if df else 0

#     def lnc_doc_calculation(self, freq):
#         return 1 + math.log10(freq) if freq > 0 else 0

#     def ltc_query_calculation(self, freq, idf):
#         return (1 + math.log10(freq)) * idf if freq > 0 else 0

#     def calculate_document_lenggth(self):
#         for doc_id in set([doc_id for term in self.dictionary for doc_id, _ in self.dictionary[term]]):
#             length = 0
#             for term, postings in self.dictionary.items():
#                 for posting_doc_id, freq in postings:
#                     if posting_doc_id == doc_id:
#                         lnc = self.lnc_doc_calculation(freq)
#                         length += lnc ** 2
#             self.document_lenggth[doc_id] = math.sqrt(length)
#         print(f"Calculated lengths for {len(self.document_lenggth)} documents")  # Debugging

#     def func_to_rank_documents(self, query):
#         query_terms = func_to_preprocess_text(query)
#         print(f"Preprocessed query terms: {query_terms}")

#         weighated_query = {}
#         query_length_norm = 0

#         for term in set(query_terms):
#             freq = query_terms.count(term)
#             idf = self.calculate_inverse_doc_freq(term)
#             ltc = self.ltc_query_calculation(freq, idf)
#             weighated_query[term] = ltc
#             query_length_norm += ltc ** 2
#             print(f"Query term '{term}': freq={freq}, idf={idf}, ltc={ltc}")  # Debugging

#         query_length_norm = math.sqrt(query_length_norm)
#         print(f"Query length: {query_length_norm}")  # Debugging

#         if query_length_norm == 0:
#             print("Query length is zero after processing.")
#             return []

#         scores = defaultdict(float)
#         for term, query_weight in weighated_query.items():
#             if term in self.dictionary:
#                 print(f"Term '{term}' found in {len(self.dictionary[term])} documents")
#                 for doc_id, freq in self.dictionary[term]:
#                     lnc = self.lnc_doc_calculation(freq)
#                     doc_length = self.document_lenggth.get(doc_id, 0)
#                     if doc_length > 0:
#                         score_contribution = (lnc * query_weight) / (doc_length * query_length_norm)
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

#         docu_ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
#         return docu_ranking[:10]

# def func_to_load_corpus_data(corpus_dir):
#     vsm = VectorSpaceModel()
#     for filename in os.listdir(corpus_dir):
#         if filename.endswith(".txt"):
#             doc_id = os.path.splitext(filename)[0]
#             document_path = os.path.join(corpus_dir, filename)
#             with open(document_path, 'r', encoding='utf-8', errors='ignore') as file:
#                 text = file.read()
#                 vsm.update_docs_in_vsm(doc_id, text)
#     vsm.calculate_document_lenggth()
#     print(f"Loaded {vsm.document_frequencyy} documents")
#     print(f"Dictionary contains {len(vsm.dictionary)} unique terms")
#     return vsm

# # Main function
# corpus_dir = r"C:\Users\ravee\Downloads\Corpus"
# vsm = func_to_load_corpus_data(corpus_dir)

# # Input query from user
# query = input("Enter your query: ")

# # Rank documents based on the query
# top_docs = vsm.func_to_rank_documents(query)

# # Output top 10 documents by relevance
# if top_docs:
#     for rank, (doc_id, score) in enumerate(top_docs, start=1):
#         print(f"{rank}. ('{doc_id}', {score:.6f})")
# else:
#     print("No matching documents found for the given query.")
