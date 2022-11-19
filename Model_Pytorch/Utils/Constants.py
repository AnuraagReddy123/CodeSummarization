

with open('vocab.txt', 'r') as f:
    vocab = eval(f.read())

VOCAB_SIZE = min(len(vocab), 10000)
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
MAX_LEN = 100
MAX_TREE_DEPTH = 10
