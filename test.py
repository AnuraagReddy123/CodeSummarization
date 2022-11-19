import nltk
from os import path

# Uncomment the below lines and run when running for the first time
# nltk.download('punkt')
# nltk.download('stopwsords')

import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
import re

import json


def _preprocess_text(text: str):

    # Remove some regex patterns

    text = re.sub(r'\\[nr]', ' ', text)
    text = re.sub(r'(^|\s)<[\w.-]+@(?=[a-z\d][^.]*\.)[a-z\d.-]*[^.]>', ' ', text)
    text = re.sub(r'https?://[-a-zA-Z0-9@:%._+~#?=/]+(?=($|[^-a-zA-Z0-9@:%._+~#?=/]))', ' ', text)
    text = re.sub(r'#[\d]+', ' ', text)
    text = re.sub(r'^(signed-off-by|co-authored-by|also-by):', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'^#+', ' ', text)
    text = re.sub(r'(^|\s|-)[\d]+(\.[\d]+){1,}', ' ', text)
    text = re.sub(r'(^|\s)[\dA-Fa-f-]{7,}(?=(\s|$))', ' ', text)
    text = re.sub(r'(^|\s|-)[\d]+(?=(\s|$))', ' ', text)

    text = text.lower()
    text_p = "".join([char if char not in string.punctuation else " " for char in text])
    
    words = word_tokenize(text_p)
    
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_words]
    
    return stemmed




text = "'Eclipse build files were missing so .eclipse project files were not being generated.\\r\\nCloses #37973\\r\\n\\r\\n'"
text = _preprocess_text(text)
# text = re.sub(r'\\[nr]', ' ', text)

print(text)

