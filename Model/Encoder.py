import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Concatenate, Bidirectional
from Utils.Structures import Node
from Utils.Utils import pad_tensor
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from Layers import TreeLSTMLayer

class Encoder(tf.keras.Model):
    '''
    Class to encode the pull requests
    '''
    def __init__(self, hidden_dim, vocab_size, embed_dim):
        '''
        Parameters:
            hidden_dim: The hidden dimension of the LSTM
            vocab_size: The size of the vocabulary
            embed_dim: The embedding dimension
        '''

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embeddings
        self.emb_commit = Embedding(vocab_size, embed_dim, name='commit_emb')
        self.emb_sc = Embedding(vocab_size, embed_dim, name='sc_emb')
        self.emb_isstitles = Embedding(vocab_size, embed_dim, name='isstitles_emb')

        # Encodings
        self.enc_commit = Bidirectional(LSTM(units=hidden_dim, return_sequences=True, return_state=True), name='commit_enc', merge_mode='sum')
        self.enc_sc = Bidirectional(LSTM(units=hidden_dim, return_sequences=True, return_state=True), name='sc_enc', merge_mode='sum')
        self.enc_isstitles = Bidirectional(LSTM(units=hidden_dim, return_sequences=True, return_state=True), name='isstitles_enc', merge_mode='sum')
        self.enc_ast = TreeLSTMLayer(hidden_dim, vocab_size, embed_dim)

        self.dense_h = Dense(hidden_dim, name='dense_h')
        self.dense_c = Dense(hidden_dim, name='dense_c')
        self.dense_mergeast_h = Dense(hidden_dim, name='dense_merge')
        self.dense_mergeast_c = Dense(hidden_dim, name='dense_merge')
        self.dense_mergeallast = Dense(1, name='dense_mergeallast')

        self.dense_merge_2 = Dense(hidden_dim, name='dense_merge_2')

    
    def call(self, batch_pr):
        pass

    def encode(self, pr):
        '''
        Parameters:
            pr: Dictionary containing comments, issue titles, 
                asts and commit messages
        '''
        commits = pr['commits'] # Contains Dictionary of commit messages

        enc_commits = []
        h_commits = []
        c_commits = []

        for commit in commits.values():
            inp_sc = commit['comments'] # Shape: (max_len,)
            inp_commit = commit['cm'] # Shape: (max_len,)
            inp_asts = [] # Shape: [old_ast, new_ast] * num_trees
            
            for old_ast, new_ast in zip(commit['old_asts'], commit['new_asts']):
                inp_asts.append([old_ast, new_ast])
            
            # Embedding
            emb_commit = self.emb_commit(inp_commit) # Shape: (max_len, embed_dim)
            emb_sc = self.emb_sc(inp_sc) # Shape: (max_len, embed_dim)

            # Encoding
            enc_commit, h_fwd, c_fwd, h_bwd, c_bwd = self.enc_commit(emb_commit) # Shape: (max_len, hidden_dim), (hidden_dim,), (hidden_dim,)
            h_commit = h_fwd + h_bwd # Shape: (hidden_dim,)
            c_commit = c_fwd + c_bwd # Shape: (hidden_dim,)
        
            enc_sc, h_fwd, c_fwd, h_bwd, c_bwd = self.enc_sc(emb_sc) # Shape: (max_len, hidden_dim), (hidden_dim,), (hidden_dim,)
            h_sc = h_fwd + h_bwd # Shape: (hidden_dim,)
            c_sc = c_fwd + c_bwd # Shape: (hidden_dim,)

            # AST Encoding
            h_asts = []
            c_asts = []
            for old_ast, new_ast in inp_asts:
                _, h_old, c_old = self.enc_ast(old_ast)
                _, h_new, c_new = self.enc_ast(new_ast)
                h = self.dense_mergeast_h(tf.concat([h_old, h_new], axis=1)) # Shape: (hidden_dim,)
                c = self.dense_mergeast_c(tf.concat([c_old, c_new], axis=1)) # Shape: (hidden_dim,)
                h_asts.append(h)
                c_asts.append(c)
            
            h_asts = tf.stack(h_asts, axis=0) # Shape: (num_trees, hidden_dim)
            c_asts = tf.stack(c_asts, axis=0) # Shape: (num_trees, hidden_dim)

            # Merge ASTs
            h_asts = self.dense_mergeallast(h_asts) # Shape: (num_trees, 1)
            c_asts = self.dense_mergeallast(c_asts) # Shape: (num_trees, 1)
            
            # Reshape to single dimension array
            h_asts = tf.reshape(h_asts, (-1,)) # Shape: (num_trees,)
            c_asts = tf.reshape(c_asts, (-1,)) # Shape: (num_trees,)

            # Concatenate all artifacts in commit
            h_commit = tf.concat([h_commit, h_sc, h_asts], axis=0) # Shape: (2*hidden_dim + num_trees,)
            c_commit = tf.concat([c_commit, c_sc, c_asts], axis=0) # Shape: (2*hidden_dim + num_trees,)

            enc_commits.append(enc_commit)
            enc_commits.append(enc_sc)
            h_commits.append(h_commit)
            c_commits.append(c_commit)

        enc_commits = tf.stack(enc_commits, axis=0) # Shape: (num_commits, max_len, hidden_dim)
        h_commits = tf.stack(h_commits, axis=0) # Shape: (num_commits, 3*hidden_dim)
        c_commits = tf.stack(c_commits, axis=0) # Shape: (num_commits, 3*hidden_dim)

        # Reduce mean
        enc_commits = tf.math.reduce_mean(enc_commits, axis=0) # Shape: (max_len, hidden_dim)
        h_commits = tf.math.reduce_mean(h_commits, axis=0) # Shape: (3*hidden_dim,)
        c_commits = tf.math.reduce_mean(c_commits, axis=0) # Shape: (3*hidden_dim,)

        inp_isstitles = pr['issue_titles'] # Shape: (max_len, )

        # Embedding
        emb_isstitles = self.emb_isstitles(inp_isstitles) # Shape: (max_len, embed_dim)

        # Encoding
        enc_isstitles, fwd_h, fwd_c, bwd_h, bwd_c = self.enc_isstitles(emb_isstitles) # Shape: (max_len, hidden_dim), (hidden_dim,), (hidden_dim,)
        h_isstitles = fwd_h + bwd_h # Shape: (hidden_dim,)
        c_isstitles = fwd_c + bwd_c # Shape: (hidden_dim,)

        # Concatenate
        h = tf.concat([h_commits, h_isstitles], axis=0) # Shape: (4*hidden_dim,)
        c = tf.concat([c_commits, c_isstitles], axis=0) # Shape: (4*hidden_dim,)
        enc = tf.concat([enc_commits, enc_isstitles], axis=0) # Shape: (2*max_len, hidden_dim)

        return enc, h, c

        
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

    root3 = Node(0)
    root3.add_child(Node(1))
    root3.add_child(Node(2))
    root3.children[0].add_child(Node(5))

    root4 = Node(0)
    root4.add_child(Node(1))
    root4.add_child(Node(2))
    root4.children[0].add_child(Node(5))

    root5 = Node(0)
    root5.add_child(Node(1))
    root5.add_child(Node(6))

    root6 = Node(0)
    root6.add_child(Node(1))
    root6.add_child(Node(6))

    root7 = Node(0)
    root7.add_child(Node(1))
    root7.add_child(Node(6))

    root8 = Node(0)
    root8.add_child(Node(1))
    root8.add_child(Node(6))

    trees = [[[root1, root2], [root3, root4]], [[root5, root6]]]
    encoder = Encoder(128, 100, 256)
    x = encoder(commit, sc, trees, isstitles)
    for i in encoder.trainable_variables:
        print(i.name, i.shape)