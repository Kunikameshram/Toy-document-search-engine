#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import nltk
import math

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter, defaultdict


# In[25]:


tf_dict = {}  #Term Frequency
df_dict = defaultdict(int) #Document Frequency
normalized_tf_idf = {} #Tf-idf
N=0 # Total number of documents


# In[26]:


corpusroot = './US_Inaugural_Addresses'
for filename in os.listdir(corpusroot):
    if filename.endswith('.txt'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close() 
        doc = doc.lower()
        
        # Tokenize the document
        tokenize = RegexpTokenizer(r'[a-zA-Z]+')
        tokens = tokenize.tokenize(doc)
        
        # print(f"before: {tokens}")
        
        #remvoving the stop words
        stop_words_list = stopwords.words('english')
        tokens_without_stopwords = [t for t in tokens if t not in stop_words_list]
        
        # print(f"after: {tokens_without_stopwords}")
        
        # Stemming on obtained tokens
        stemmer = PorterStemmer()
        final_tokens = [stemmer.stem(token) for token in tokens_without_stopwords]
        
        # Term frequency for document
        tf_dict[filename] = Counter(final_tokens)
        
        # Document Frequency for unique tokens
        unique_tokens = set(final_tokens)  
        for token in unique_tokens:
            df_dict[token] += 1
        
        N+=1


# In[27]:


def calculate_idf(token):
    return math.log10(N / df_dict[token]) if df_dict[token] > 0 else -1


# In[46]:


def getidf(token):
    stemmed_token = stemmer.stem(token)
    return calculate_idf(stemmed_token)


# In[29]:


# Get tf-idf value 
def get_tf_idf_weight(filename,token):
    if(tf_dict[filename][token])!=0:
        return (1+math.log10(tf_dict[filename][token]))*calculate_idf(token)
    else:
        return 0


# In[30]:


# Normalize document vector
def normalize(weights):
    length = math.sqrt(sum(weight ** 2 for weight in weights.values()))
    return {token: weight / length for token, weight in weights.items()}


# In[31]:


# Build normalized TF-IDF vectors
for filename in tf_dict:
    tf_idf_weight = {token: get_tf_idf_weight(filename, token) for token in tf_dict[filename]}
    normalized_tf_idf[filename] = normalize(tf_idf_weight)
        


# In[32]:


# Retrieve normalized TF-IDF weight for a term in a document
def return_weight(filename, token):
    return normalized_tf_idf[filename].get(token, 0)   


# In[33]:


# Construction of postings list
list_sorted_tf_idf = {}

# list for each token (sorted by TF-IDF weights)
for filename, tfidf_vector in normalized_tf_idf.items():
    for token, weight in tfidf_vector.items():
        if token not in list_sorted_tf_idf:
            list_sorted_tf_idf[token] = []
        list_sorted_tf_idf[token].append((filename, weight))


# In[34]:


for token in list_sorted_tf_idf:
    list_sorted_tf_idf[token].sort(key=lambda x: x[1], reverse=True)


# In[60]:


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

    # Cosine similarity
    for token in stemmed_query_tokens:
        if token in list_sorted_tf_idf:
            top_10_postings = list_sorted_tf_idf[token][:10]  # top-10 elements
            # print(f"Token={token} Posting= {top_10_postings}")
            
            # 10th weight is the upper-bound weight 
            upper_bound_weight = top_10_postings[-1][1] if len(top_10_postings) == 10 else 0
            
            for doc, weight in top_10_postings:
                actual_scores[doc] += query_tf[token] * weight / query_magnitude
            
            # Document not in top-10 get the weight as upper-bound)
            for doc in normalized_tf_idf:
                if doc not in actual_scores:
                    upper_bound_scores[doc] += query_tf[token] * upper_bound_weight / query_magnitude
        else:
            # If no document contains any token in the query return None
            return ("None", 0)
    
    # If actual_scores is empty, return None
    if not actual_scores:
        return ("None", 0)
    
    # Merge actual and upper-bound scores
    for doc in normalized_tf_idf:
        if doc not in actual_scores:
            actual_scores[doc] = upper_bound_scores[doc]
    
    # Finding the document with the highest actual score
    best_doc = max(actual_scores.items(), key=lambda x: x[1])
    
# If the best document is in upper-bound scores, fetch more elements
    if best_doc[0] in upper_bound_scores:
        return ("fetch more", 0)
    else:
    # If it's not, return the best document and its score
        return best_doc



# In[65]:


def getweight(filename, token):
    stemmed_token = stemmer.stem(token)
    return return_weight(filename, stemmed_token)


# In[68]:

# Tests given
print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt','leadership'))
print("--------------")
print("(%s, %.12f)" % query("states laws"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("world civilization"))


# In[47]:




