MAX_VOCAB_SIZE = 10000

with open('Data/vocab.txt', 'r') as f:
    vocab = eval(f.read())

VOCAB_SIZE = min(len(vocab), MAX_VOCAB_SIZE)
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
MAX_COMMITS = 10
MAX_LEN = 100
MAX_TREE_DEPTH = 10
EPOCHS = 100
BATCH_SIZE = 16
NUM_LAYERS = 3
