import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Concatenate
from Attention import Attention
import keras
import os
from Encoder import Encoder
from Utils.Structures import Node
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Decoder(keras.Model):
    def __init__(self, hidden_dim, vocab_size, embed_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.prdesc_emb = Embedding(vocab_size, embed_dim, name='prdesc_emb')
        self.attention = Attention(hidden_dim)
        self.prdesc_enc = LSTM(units=hidden_dim, return_sequences=True, return_state=True, name='prdesc_enc')

        self.Wc = Dense(hidden_dim, activation='tanh', name='Wc')
        self.fc = Dense(vocab_size, name='fc')
    
    def call(self, prdesc, enc_output, enc_mask, state=None):
        # seq_len = decoder
        # max_len = encoder
        # prdesc: (batch_size, seq_len)
        # enc_output: (batch_size, max_len, hidden_dim)
        # hidden: (batch_size, hidden_dim)
        # cell: (batch_size, hidden_dim)
        # state: [(batch_size, hidden_dim), (batch_size, hidden_dim)]
        # enc_mask: (batch_size, max_len)

        # Embedding
        prdesc_emb = self.prdesc_emb(prdesc) # Shape (batch_size, seq_len, embed_dim)
        
        # Processs one step
        prdesc_enc, h, c = self.prdesc_enc(prdesc_emb, initial_state=state) 
        # Shape (batch_size, seq_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)

        # Use lstm output as query for attention
        context_vector, attention_weights = self.attention(query=prdesc_enc, value=enc_output, mask=enc_mask)
        # Shape (batch_size, seq_len, hidden_dim), (batch_size, seq_len, max_len)
        context_and_pdesc_enc = tf.concat([context_vector, prdesc_enc], axis=-1)

        # at = tanh(Wc@[ct; ht])
        attention_vector = self.Wc(context_and_pdesc_enc) # Shape (batch_size, seq_len, hidden_dim)

        logits = self.fc(attention_vector) # Shape (batch_size, seq_len, vocab_size)

        return logits, attention_weights, h, c


if __name__ == '__main__':
    commit = tf.random.uniform((2, 100), minval=0, maxval=100, dtype=tf.int32)
    sc = tf.random.uniform((2, 100), minval=0, maxval=100, dtype=tf.int32)
    isstitles = tf.random.uniform((2, 100), minval=0, maxval=100, dtype=tf.int32)

    root1 = Node(0)
    root1.add_child(Node(1))
    root1.add_child(Node(2))
    root1.children[0].add_child(Node(3))
    root1.children[0].add_child(Node(4))

    root2 = Node(0)
    root2.add_child(Node(1))
    root2.add_child(Node(2))

    trees = [root1, root2]
    encoder = Encoder(128, 100, 256)
    enc, h, c, mask = encoder(commit, sc, trees, isstitles)

    decoder = Decoder(128, 100, 256)
    prdesc = tf.random.uniform((2, 10), minval=0, maxval=100, dtype=tf.int32)
    print(prdesc)
    inp = tf.constant([[2], [2]])
    logits, attention_weights, h, c = decoder(inp, enc, mask, state=[h, c])
    print(logits.shape)
    print(attention_weights.shape)
    print(h.shape)
    print(c.shape)