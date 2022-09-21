import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Concatenate
import keras
import os
from Encoder import Encoder
from Utils.Structures import Node
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Decoder(tf.keras.Model):
    '''
    Class to decode the encodings
    '''
    def __init__(self, hidden_dim, vocab_size, embed_dim):
        '''
        Parameters:
            hidden_dim: The hidden dimension of the LSTM
            vocab_size: The size of the vocabulary
            embed_dim: The embedding dimension
        '''

        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embeddings
        self.emb_prdesc = Embedding(vocab_size, embed_dim, name='prdesc_emb')

        # Decodings
        self.dec_prdesc = LSTM(units=hidden_dim, return_sequences=True, return_state=True, name='prdesc_dec')

        # Merge
        self.fc = Dense(vocab_size, name='fc')
    
    def call(self, batch_prdesc, h_enc, c_enc):
        '''
        Parameters:
            batch_pr : Batch of prs
                Shape: (batch_size, max_pr_len)
        '''
        # Get the embeddings
        emb_prdesc = self.emb_prdesc(batch_prdesc) # Shape: (batch_size, max_pr_len, embed_dim)

        # Decode
        dec_prdesc, h_dec, c_dec = self.dec_prdesc(emb_prdesc, initial_state=[h_enc, c_enc]) # Shape: (batch_size, max_pr_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)

        # Merge
        logits = self.fc(dec_prdesc) # Shape: (batch_size, max_pr_len, vocab_size)

        return logits, h_dec, c_dec
        


if __name__ == '__main__':
    # Generate random data
    pr = {}
    pr['issue_titles'] = np.random.randint(0, 100, (100, ))
    pr['commits'] = {}

    for i in range(10):
        pr['commits'][i] = {}
        pr['commits'][i]['cm'] = np.random.randint(0, 100, (100, ))
        pr['commits'][i]['comments'] = np.random.randint(0, 100, (100, ))
        
        pr['commits'][i]['old_asts'] = []
        pr['commits'][i]['new_asts'] = []

        for j in range(5):
            root = Node(0)
            root.add_child(Node(1))
            root.add_child(Node(2))

            pr['commits'][i]['old_asts'].append(root)

            root = Node(0)
            root.add_child(Node(1))

            pr['commits'][i]['new_asts'].append(root)
        
    # Create model
    encoder = Encoder(128, 150, 256)
    pr2 = pr.copy()
    h, c = encoder([pr, pr2])
    print(h.shape)
    print(c.shape)

    decoder = Decoder(128, 150, 256)
    prdesc = np.random.randint(0, 100, (2, 100))
    fc, h_dec, c_dec = decoder(prdesc, h, c)

    print(fc.shape)
    print(h_dec.shape)
    print(c_dec.shape)
