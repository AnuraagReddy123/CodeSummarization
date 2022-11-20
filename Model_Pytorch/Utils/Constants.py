

with open('vocab.txt', 'r') as f:
    vocab = eval(f.read())

VOCAB_SIZE = min(len(vocab), 10000)
EMBEDDING_DIM = 512
HIDDEN_DIM = 512
MAX_COMMITS = 10
MAX_LEN = 100
MAX_TREE_DEPTH = 10
EPOCHS = 30
BATCH_SIZE = 1
NUM_LAYERS = 5
