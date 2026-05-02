import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # <--- 1. Thêm thư viện Stemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Khởi tạo công cụ cắt tỉa từ gốc
stemmer = PorterStemmer() # <--- 2. Khởi tạo Stemmer

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
              'treatment', 'associated', 'patients', 'may','day', 'case','old','u','n','ive','ate','brother','dad','basic','im'])

negation_words = {
    "no", "not", "nor", "none", "never", 
    "isn", "aren", "wasn", "weren", 
    "hasn", "haven", "hadn", 
    "doesn", "don", "didn", 
    "won", "wouldn", "shan", "shouldn", 
    "can", "couldn", "mustn"
}

STOP_WORDS = STOP_WORDS - negation_words

def expand_contractions(text):
    text = text.lower()
    text = text.replace("don't", "do not")
    text = text.replace("doesn't", "does not")
    text = text.replace("didn't", "did not")
    text = text.replace("isn't", "is not")
    text = text.replace("aren't", "are not")
    text = text.replace("wasn't", "was not")
    text = text.replace("weren't", "were not")
    text = text.replace("can't", "can not")
    text = text.replace("couldn't", "could not")
    text = text.replace("won't", "will not")
    return text

def normalize(text):
    text = str(text).lower()
    text = expand_contractions(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 3. Thêm tham số use_stemming=True
def tokenize(text, remove_stopwords=True, use_bigrams=True, use_stemming=True):
    text = normalize(text)
    tokens = text.split()

    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    
    # 4. THỰC HIỆN CẮT TỈA TỪ GỐC (STEMMING) TRƯỚC KHI TẠO BIGRAM
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    
    final_tokens = list(tokens)

    if use_bigrams and len(tokens) > 1:
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            final_tokens.append(bigram)
            
    return final_tokens