from collections import Counter
from preprocess import tokenize
import math
import scipy.sparse as sp

def build_vocabulary(texts, min_freq=1,remove_stopwords=True,use_bigrams=True):
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

def texts_to_matrix(texts,vocab_obj,remove_stopwords=True,use_bigrams=True):
    word2idx=vocab_obj["word2idx"]
    idf_dict=vocab_obj["idf"]

    row_indices=[]
    col_indices=[]
    data_values=[]

    for row_idx, text in enumerate(texts):
        tokens= tokenize(text,remove_stopwords=remove_stopwords,use_bigrams=use_bigrams)
        total_words=len(tokens)
        if total_words==0:
            continue
    
        tf_counts=Counter(tokens)

        for word,count in tf_counts.items():
            if word in word2idx:
                tf=count/total_words
                idf=idf_dict[word]
                
                row_indices.append(row_idx)
                col_indices.append(word2idx[word])
                data_values.append(tf*idf)
    sparse_matrix=sp.csr_matrix(
        (data_values,(row_indices,col_indices)),
        shape=(len(texts),len(word2idx))
    )
    return sparse_matrix.tocsc()
    


def text_to_vector(text,vocab_obj,remove_stopwords=True,use_bigrams=True):
    return texts_to_matrix([text],vocab_obj,remove_stopwords,use_bigrams)