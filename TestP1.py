import os
import nltk
import math

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter, defaultdict

tf_dict = {}  #Term Frequency
df_dict = defaultdict(int)
N=0 # total number of 
normalized_tf_idf = {}
stemmer = PorterStemmer()

# reading the documents
corpusroot = './US_Inaugural_Addresses'
for filename in os.listdir(corpusroot):
    if filename.endswith('.txt'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close() 
        doc = doc.lower()
        
        # tokenize the document
        tokenize = RegexpTokenizer(r'[a-zA-Z]+')
        tokens = tokenize.tokenize(doc)
        
        # print(f"before: {tokens}")
        
        #remvoving the stop words
        stop_words_list = stopwords.words('english')
        tokens_without_stopwords = [t for t in tokens if t not in stop_words_list]
        
        # print(f"after: {tokens_without_stopwords}")
        
        # Stemming on obtained tokens
        final_tokens = [stemmer.stem(token) for token in tokens_without_stopwords]
        
        # Term frequency for document
        tf_dict[filename] = Counter(final_tokens)
        
        # Document Frequency for unique tokens
        unique_tokens = set(final_tokens)  
        for token in unique_tokens:
            df_dict[token] += 1
        
        N+=1

# Function to calculate IDF
def calculate_idf(token):
    return math.log10(N / df_dict[token]) if df_dict[token] > 0 else -1

# Function to get IDF
def get_idf(token):
    stemmed_token = stemmer.stem(token)
    return calculate_idf(stemmed_token)

# get tf-idf value 
def get_tf_idf_weight(filename,token):
    if(tf_dict[filename][token])!=0:
        return (1+math.log10(tf_dict[filename][token]))*calculate_idf(token)
    else:
        return 0

# Normalize document vector
def normalize(weights):
    length = math.sqrt(sum(weight ** 2 for weight in weights.values()))
    return {token: weight / length for token, weight in weights.items()}

# Build normalized TF-IDF vectors
for filename in tf_dict:
    tf_idf_weight = {token: get_tf_idf_weight(filename, token) for token in tf_dict[filename]}
    normalized_tf_idf[filename] = normalize(tf_idf_weight)
        
# Retrieve normalized TF-IDF weight for a term in a document
def return_weight(filename, token):
    return normalized_tf_idf[filename].get(token, 0)      
       
# Construction of postings list
posting_list = {}

# list for each token (sorted by TF-IDF weights)
for filename, tfidf_vector in normalized_tf_idf.items():
    for token, weight in tfidf_vector.items():
        if token not in posting_list:
            posting_list[token] = []
        posting_list[token].append((filename, weight))
        
for token in posting_list:
    posting_list[token].sort(key=lambda x: x[1], reverse=True)
    
def query(qstring):
    query_tokens = qstring.lower().split()
    stemmed_query_tokens = [stemmer.stem(token) for token in query_tokens]
    
    query_tf = {}
    length = 0
    
    # Calculate query term frequency and magnitute
    for token in stemmed_query_tokens:
        if token not in query_tf:
            query_tf[token] = 1 + math.log10(stemmed_query_tokens.count(token))
        length += query_tf[token] ** 2
    
    query_magnitude = math.sqrt(length)
    
    actual_scores = Counter() #this stores the cosine similarity of complete matches
    upper_bound_scores = Counter() #this stores cosine similarity value of partial match and upper bound

    # cosine similarity
    for token in stemmed_query_tokens:
        if token in posting_list:
            top_10_postings = posting_list[token][:10]  # top-10 elements
            print(f"Token={token} Posting= {top_10_postings}")
            # 10th weight is the upper-bound weight 
            upper_bound_weight = top_10_postings[-1][1] if len(top_10_postings) == 10 else 0
            
            for doc, weight in top_10_postings:
                actual_scores[doc] += query_tf[token] * weight / query_magnitude
            
            # document not in top-10 get the weight as upper-bound)
            for doc in normalized_tf_idf:
                if doc not in actual_scores:
                    upper_bound_scores[doc] += query_tf[token] * upper_bound_weight / query_magnitude
        else:
            # Token doesn't exist in the corpus; ignore it
            continue
    
    # If actual_scores is empty, return None
    if not actual_scores:
        return ("None", 0)
    
    # Merge actual and upper-bound scores (if needed)
    for doc in normalized_tf_idf:
        if doc not in actual_scores:
            actual_scores[doc] = upper_bound_scores[doc]
    
    # Find the document with the highest actual score
    best_doc = max(actual_scores.items(), key=lambda x: x[1])
    
# Check if the best_doc is in upper_bound_scores
    if best_doc[0] in upper_bound_scores:
    # If the best document is in upper-bound scores, fetch more elements
        return ("fetch more", 0)
    else:
    # If it's not, return the best document and its score
        return best_doc

