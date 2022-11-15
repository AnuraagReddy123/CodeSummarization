import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Concatenate, Bidirectional
from Utils.Structures import Node
from Utils.Utils import pad_tensor
import keras
import numpy as np
import os
from Layers import TreeLSTMLayer
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Encoder(tf.keras.Model):
    '''
    Class to encode the pull requests
    Fix number of commits and trees
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
        self.enc_commit = Bidirectional(LSTM(units=hidden_dim, return_sequences=True, return_state=True), name='commit_enc', merge_mode='concat')
        self.enc_sc = Bidirectional(LSTM(units=hidden_dim, return_sequences=True, return_state=True), name='sc_enc', merge_mode='concat')
        self.enc_isstitles = Bidirectional(LSTM(units=hidden_dim, return_sequences=True, return_state=True), name='isstitles_enc', merge_mode='concat')
        self.enc_ast = TreeLSTMLayer(hidden_dim, vocab_size, embed_dim)

        self.dense_h = Dense(hidden_dim, name='dense_h')
        self.dense_c = Dense(hidden_dim, name='dense_c')
        self.dense_mergeast_h = Dense(hidden_dim, name='dense_merge')
        self.dense_mergeast_c = Dense(hidden_dim, name='dense_merge')
        self.dense_mergeallast = Dense(1, name='dense_mergeallast')
        self.dense_mergecommits = Dense(1, name='dense_mergecommits')
        self.dense_mergeh = Dense(hidden_dim, name='dense_mergeh')
        self.dense_mergec = Dense(hidden_dim, name='dense_mergec')

    
    def call(self, batch_pr):
        '''
        Parameters:
            batch_pr : Batch of prs
        '''
        enclist = []
        hlist = []
        clist = []

        for pr in batch_pr:
            h, c = self.encode(pr)
            # enclist.append(enc)
            hlist.append(h)
            clist.append(c)

        
        hlist = tf.stack(hlist)
        clist = tf.stack(clist)

        return hlist, clist

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
            
            for old_ast, new_ast in zip(commit['old_asts'], commit['cur_asts']):
                inp_asts.append([old_ast, new_ast])
            
            # Embedding
            emb_commit = self.emb_commit(inp_commit) # Shape: (max_len, embed_dim)
            emb_sc = self.emb_sc(inp_sc) # Shape: (max_len, embed_dim)

            # Increase dim
            emb_commit = tf.expand_dims(emb_commit, axis=0) # Shape: (1, max_len, embed_dim)
            emb_sc = tf.expand_dims(emb_sc, axis=0) # Shape: (1, max_len, embed_dim)

            # Encoding
            enc_commit, h_fwd, c_fwd, h_bwd, c_bwd = self.enc_commit(emb_commit) # Shape: (1, max_len, 2*hidden_dim), (1, hidden_dim), (1, hidden_dim), (1, hidden_dim), (1, hidden_dim)
            h_commit = Concatenate()([h_fwd, h_bwd]) # Shape: (1, 2*hidden_dim)
            c_commit = Concatenate()([c_fwd, c_bwd]) # Shape: (1, 2*hidden_dim)
        
            enc_sc, h_fwd, c_fwd, h_bwd, c_bwd = self.enc_sc(emb_sc) # Shape: (1, max_len, 2*hidden_dim), (1, hidden_dim), (1, hidden_dim), (1, hidden_dim), (1, hidden_dim)
            h_sc = Concatenate()([h_fwd, h_bwd]) # Shape: (1, 2*hidden_dim)
            c_sc = Concatenate()([c_fwd, c_bwd]) # Shape: (1, 2*hidden_dim)

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

            # Increase dim
            h_asts = tf.expand_dims(h_asts, axis=0) # Shape: (1, num_trees)
            c_asts = tf.expand_dims(c_asts, axis=0) # Shape: (1, num_trees)

            # Concatenate all artifacts in commit
            h_commit = Concatenate()([h_commit, h_sc, h_asts]) # Shape: (1, 2*hidden_dim + num_trees)
            c_commit = Concatenate()([c_commit, c_sc, c_asts]) # Shape: (1, 2*hidden_dim + num_trees)

            # Remove dim
            h_commit = tf.squeeze(h_commit, axis=0) # Shape: (2*hidden_dim + num_trees,)
            c_commit = tf.squeeze(c_commit, axis=0) # Shape: (2*hidden_dim + num_trees,)

            # enc_commits.append(enc_commit)
            # enc_commits.append(enc_sc)
            h_commits.append(h_commit)
            c_commits.append(c_commit)

        # enc_commits = tf.stack(enc_commits, axis=0) # Shape: (num_commits, max_len, hidden_dim)
        h_commits = tf.stack(h_commits, axis=0) # Shape: (num_commits, 4*hidden_dim + num_trees)
        c_commits = tf.stack(c_commits, axis=0) # Shape: (num_commits, 4*hidden_dim + num_trees)

        # Merge commits
        h_commits = self.dense_mergecommits(h_commits) # Shape: (num_commits, 1)
        c_commits = self.dense_mergecommits(c_commits) # Shape: (num_commits, 1)

        # Reshape to single dimension array
        h_commits = tf.reshape(h_commits, (-1,)) # Shape: (num_commits,)
        c_commits = tf.reshape(c_commits, (-1,)) # Shape: (num_commits,)

        # Increase dim
        h_commits = tf.expand_dims(h_commits, axis=0) # Shape: (1, num_commits)
        c_commits = tf.expand_dims(c_commits, axis=0) # Shape: (1, num_commits)

        # Reduce mean
        # enc_commits = tf.math.reduce_mean(enc_commits, axis=0) # Shape: (max_len, hidden_dim)

        inp_isstitles = pr['issue_title'] # Shape: (max_len, )

        # Embedding
        emb_isstitles = self.emb_isstitles(inp_isstitles) # Shape: (max_len, embed_dim)

        # Increase dim
        emb_isstitles = tf.expand_dims(emb_isstitles, axis=0) # Shape: (1, max_len, embed_dim)

        # Encoding
        enc_isstitles, fwd_h, fwd_c, bwd_h, bwd_c = self.enc_isstitles(emb_isstitles) # Shape: (1, max_len, 2*hidden_dim), (1, hidden_dim), (1, hidden_dim), (1, hidden_dim), (1, hidden_dim)
        h_isstitles = fwd_h + bwd_h # Shape: (1, hidden_dim)
        c_isstitles = fwd_c + bwd_c # Shape: (1, hidden_dim)

        # Concatenate
        h = Concatenate()([h_commits, h_isstitles]) # Shape: (1, 2*hidden_dim + num_commits)
        c = Concatenate()([c_commits, c_isstitles]) # Shape: (1, 2*hidden_dim + num_commits)
        
        # enc = tf.concat([enc_commits, enc_isstitles], axis=0) # Shape: (2*max_len, hidden_dim)

        # Merge
        h = self.dense_mergeh(h) # Shape: (1, hidden_dim)
        c = self.dense_mergec(c) # Shape: (1, hidden_dim)

        # Remove dim
        h = tf.squeeze(h, axis=0) # Shape: (hidden_dim,)
        c = tf.squeeze(c, axis=0) # Shape: (hidden_dim,)
        
        return h, c

if __name__ == '__main__':
    # Generate random data
    pr = {}
    pr['issue_title'] = np.random.randint(0, 100, (85, ))
    pr['commits'] = {}

    for i in range(10):
        pr['commits'][i] = {}
        pr['commits'][i]['cm'] = np.random.randint(0, 100, (100, ))
        pr['commits'][i]['comments'] = np.array([2])
        
        pr['commits'][i]['old_asts'] = []
        pr['commits'][i]['cur_asts'] = []

        for j in range(5):
            root = Node(0, 0)
            root.add_child(Node(1, 1))
            root.add_child(Node(2, 2))

            pr['commits'][i]['old_asts'].append(root)

            root = Node(0, 0)
            root.add_child(Node(1, 1))

            pr['commits'][i]['cur_asts'].append(root)
        
    # Create model
    encoder = Encoder(128, 150, 256)
    pr2 = pr.copy()
    h, c = encoder([pr, pr2])
    print(h.shape)
    print(c.shape)