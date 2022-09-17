import tensorflow as tf
import keras
from Utils.Structures import Node
from keras.layers import Embedding, LSTM, Dense, Concatenate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class TreeLSTMLayer(tf.keras.Model):
    def __init__(self, hidden_dim, vocab_size, embed_dim):
        super(TreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.emb = Embedding(vocab_size, embed_dim)

        self.Wi = Dense(hidden_dim, use_bias=False)
        self.Ui = Dense(hidden_dim, use_bias=False)
        self.bi = self.add_weight(shape=(hidden_dim,), initializer='zeros', name='bi', trainable=True)

        self.Wf = Dense(hidden_dim, use_bias=False)
        self.Uf = Dense(hidden_dim, use_bias=False)
        self.bf = self.add_weight(shape=(hidden_dim,), initializer='zeros', name='bf', trainable=True)

        self.Wo = Dense(hidden_dim, use_bias=False)
        self.Uo = Dense(hidden_dim, use_bias=False)
        self.bo = self.add_weight(shape=(hidden_dim,), initializer='zeros', name='bo', trainable=True)

        self.Wu = Dense(hidden_dim, use_bias=False)
        self.Uu = Dense(hidden_dim, use_bias=False)
        self.bu = self.add_weight(shape=(hidden_dim,), initializer='zeros', name='bu', trainable=True)
    
    def call(self, root:Node):
        def recurse(node:Node, hidden_states:list):
            node.emb = self.emb(tf.constant([node.id]))
            if len(node.children) != 0:
                child_h = []
                child_c = []
                for child in node.children:
                    states = recurse(child, hidden_states)
                    child_h.append(states[0])
                    child_c.append(states[1])
                
                child_h = tf.concat(child_h, axis=0)
                child_c = tf.concat(child_c, axis=0)

                # Add all children hidden states
                h_j_tilda = tf.reduce_sum(child_h, axis=0)
                h_j_tilda = tf.reshape(h_j_tilda, [1, -1])

                # Input gate
                i_j = tf.sigmoid(self.Wi(node.emb) + self.Ui(h_j_tilda) + self.bi)

                # Forget gate for each child hidden state in child_h
                f_j = []
                for i in range(len(child_h)):
                    temp = tf.reshape(child_h[i], [1, -1])
                    f_j.append(tf.sigmoid(self.Wf(node.emb) + self.Uf(temp) + self.bf))
                f_j = tf.concat(f_j, axis=0)

                # Output gate
                o_j = tf.sigmoid(self.Wo(node.emb) + self.Uo(h_j_tilda) + self.bo)

                # Update gate
                u_j = tf.tanh(self.Wu(node.emb) + self.Uu(h_j_tilda) + self.bu)

                # Cell state
                c_j = tf.multiply(i_j, u_j) + tf.reduce_sum(tf.multiply(f_j, child_c), axis=0)
                c_j = tf.reshape(c_j, [1, -1])

                # Hidden state
                h_j = tf.multiply(o_j, tf.tanh(c_j))

                node.h = h_j
                node.c = c_j
                hidden_states.append(h_j)

                return h_j, c_j
            else:
                node.h = tf.zeros([1, self.hidden_dim])
                node.c = tf.zeros([1, self.hidden_dim])
                
                hidden_states.append(node.h)
                return node.h, node.c
        
        hidden_states = []
        recurse(root, hidden_states)
        hidden_states = tf.concat(hidden_states, axis=0)
        return hidden_states, root.h, root.c
        

if __name__ == "__main__":
    root = Node(0)
    root.add_child(Node(1))
    root.add_child(Node(2))

    root.children[0].add_child(Node(3))
    root.children[0].add_child(Node(4))
    root.children[1].add_child(Node(5))

    layer = TreeLSTMLayer(5, 100, 10)
    out, h, c = layer(root)
    print(out.shape)
    print(h.shape)
    print(c.shape)
    for i in layer.trainable_variables:
        print(i.name, i.shape)

