from collections import Counter
from preprocess import tokenize
import math

def build_vocabulary(texts, min_freq=1,remove_stopwords=True):
    doc_counts = Counter()
    total_docs=len(texts)

    for text in texts:
        tokens = set(tokenize(text,remove_stopwords=remove_stopwords))
        doc_counts.update(tokens)
    
    vocab={"word2idx":{},"idf":{}}
    index=0

    for word, df in doc_counts.items():
        if df >=min_freq:
            vocab["word2idx"][word]=index
            vocab["idf"][word]=math.log(total_docs/(1+df))
            index+=1
    return vocab

def text_to_vector(text,vocab_obj,remove_stopwords=True):
    word2idx=vocab_obj["word2idx"]
    idf_dict=vocab_obj["idf"]
    vector=[0.0]*len(word2idx)
    tokens= tokenize(text,remove_stopwords=remove_stopwords)

    total_words=len(tokens)
    if total_words==0:
        return vector
    
    tf_counts=Counter(tokens)

    for word,count in tf_counts.items():
        if word in word2idx:
            tf=count/total_words
            idf=idf_dict[word]
            idx=word2idx[word]
            vector[idx]=tf*idf
    
    return vector

def texts_to_matrix(texts,vocab_obj,remove_stopwords=True):
    return [text_to_vector(text,vocab_obj,remove_stopwords=remove_stopwords) for text in texts]
