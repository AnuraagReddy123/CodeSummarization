'''
Preprocesses all the text in the dataset.
Replaces all the words in the text (including label in ASTs) with integers by computing vocabulary.
Removes some unnecessary fields in the data points like "id", "cms".
'''

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

import json

MAX_VOCAB = 5000

def _preprocess_text(text: str):

    text = text.lower()
    text_p = "".join([char if char not in string.punctuation else " " for char in text])
    
    words = word_tokenize(text_p)
    
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_words]
    
    return stemmed


def preprocess_text(dataset: dict):

    for key in dataset:

        dataset[key]['body'] = ' '.join(_preprocess_text(dataset[key]['body']))
        
        del dataset[key]['id']
        del dataset[key]['cms']
        # cms = []
        # for cm in dataset[key]['cms']:
        #     cms.append(' '.join(_preprocess_text(cm)))
        # dataset[key]['cms'] = cms

        i = 0
        for commit_sha in dataset[key]['commits']:

            # dataset[key]['commits'][commit]['cm'] = cms[i]
            cm = dataset[key]['commits'][commit_sha]['cm']
            dataset[key]['commits'][commit_sha]['cm'] = ' '.join(_preprocess_text(cm))
            i += 1

            comment_para = ' '.join(dataset[key]['commits'][commit_sha]['comments'])
            comment_para = ' '.join(_preprocess_text(comment_para))
            dataset[key]['commits'][commit_sha]['comments'] = comment_para
            
        dataset[key]['issue_title'] = ' '.join(_preprocess_text(dataset[key]['issue_title']))
    
    return dataset


def compute_vocab(dataset: dict):

    vocab_dict = {}

    def _add(word):
        if word in vocab_dict:
            vocab_dict[word] += 1
        else:
            vocab_dict[word] = 1

    for key in dataset:

        for x in dataset[key]['body'].split():
            _add(x)
        for x in dataset[key]['issue_title'].split():
            _add(x)

        # vocab_set = vocab_set.union(dataset[key]['body'].split())
        # vocab_set = vocab_set.union(dataset[key]['issue_title'].split())

        for commit_sha in dataset[key]['commits']:
            # vocab_set = vocab_set.union(dataset[key]['commits'][commit_sha]['cm'].split())
            # vocab_set = vocab_set.union(dataset[key]['commits'][commit_sha]['comments'].split())

            for x in dataset[key]['commits'][commit_sha]['cm'].split():
                _add(x)
            for x in dataset[key]['commits'][commit_sha]['comments'].split():
                _add(x)

            old_asts = dataset[key]['commits'][commit_sha]['old_asts']
            new_asts = dataset[key]['commits'][commit_sha]['new_asts']

            for old_ast in old_asts:
                for node_id in old_ast:
                    _add(old_ast[node_id]['label'])
                    # vocab_set.add(old_ast[node_id]['label'])

            for new_ast in new_asts:
                for node_id in new_ast:
                    _add(new_ast[node_id]['label'])
                    # vocab_set.add(new_ast[node_id]['label'])

    # <START> -> 0
    # <BLANK> -> 1
    # <END> -> 2
    # <UNK> -> 3
    
    vocab_count = list(vocab_dict.items())
    vocab_count.sort(key=lambda k: k[1], reverse=True)
    if len(vocab_count) > MAX_VOCAB-4:
        vocab_count = vocab_count[:MAX_VOCAB-4]

    vocab = [x[0] for x in vocab_count]

    vocab = ['<START>', '<BLANK>', '<END>', '<UNK>'] + vocab

    return vocab



def encode_word_to_index(dataset: dict, vocab: list):

    def _index(word):
        if word in vocab:
            return vocab.index(word)
        else:
            return 3

    for key in dataset:

        dataset[key]['body'] = [_index(x) for x in dataset[key]['body'].split()]
        dataset[key]['issue_title'] = [_index(x) for x in dataset[key]['issue_title'].split()]

        for commit_sha in dataset[key]['commits']:
            cm = dataset[key]['commits'][commit_sha]['cm']
            dataset[key]['commits'][commit_sha]['cm'] = [_index(x) for x in cm.split()]

            comments = dataset[key]['commits'][commit_sha]['comments']
            dataset[key]['commits'][commit_sha]['comments'] = [_index(x) for x in comments.split()]

            old_asts = dataset[key]['commits'][commit_sha]['old_asts']
            new_asts = dataset[key]['commits'][commit_sha]['new_asts']

            for old_ast in old_asts:
                for node_id in old_ast:
                    old_ast[node_id]['label'] = _index(old_ast[node_id]['label'])

            for new_ast in new_asts:
                for node_id in new_ast:
                    new_ast[node_id]['label'] = _index(new_ast[node_id]['label'])

    return dataset



if __name__=='__main__':

    with open('../Data/dataset_aug.json') as f:
        dataset: dict = json.load(f)

    dataset = preprocess_text(dataset)

    if not path.exists('vocab.txt'):
        vocab: list = compute_vocab(dataset)
        with open('vocab.txt', 'w+') as f:
            f.write(str(vocab))

    with open('vocab.txt', 'r') as f:
        vocab = eval(f.read())

    dataset = encode_word_to_index(dataset, vocab)

    with open('../Data/dataset_preproc.json', 'w+') as f:
        f.write(json.dumps(dataset))

