import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
              'treatment', 'associated', 'patients', 'may','day', 'case','old','u','n','didnt','ive','ate','feel','keep'
                ,'brother','dad','basic','im'])

negation_words = {
    "no", "not", "nor", "none", "never", 
    "isn", "aren", "wasn", "weren", 
    "hasn", "haven", "hadn", 
    "doesn", "don", "didn", 
    "won", "wouldn", "shan", "shouldn", 
    "can", "couldn", "mustn"
}

STOP_WORDS=STOP_WORDS-negation_words

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text, remove_stopwords=True,use_bigrams=True):
    text = normalize(text)
    tokens= text.split()

    if remove_stopwords:
        tokens=[token for token in tokens if token not in STOP_WORDS]
    
    final_tokens=list(tokens)

    if use_bigrams and len(tokens)>1:
        for i in range(len(tokens)-1):
            bigram=f"{tokens[i]}_{tokens[i+1]}"
            final_tokens.append(bigram)
    return final_tokens