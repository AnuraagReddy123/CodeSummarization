import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class Attention(tf.keras.layers.Layer):
    def __init__ (self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, name='W1', use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, name='W2', use_bias=False)
        
        self.attention = tf.keras.layers.AdditiveAttention(name='attention')

    def call(self, query, value, mask):
        '''
        Shapes
        query: (batch_size, dec_len, hidden_dim)
        value: (batch_size, enc_len, hidden_dim)
        mask: (batch_size, enc_len)
        '''

        w1_query = self.W1(query) # Shape (batch_size, dec_len, attn_units)
        w2_key = self.W2(value) # Shape (batch_size, enc_len, attn_units)

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask
    
        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores = True,
        )

        '''
        Shapes
        context_vector: (batch_size, dec_len, hidden_dim)
        attention_weights: (batch_size, dec_len, enc_len)
        '''

        return context_vector, attention_weights
        
    
if __name__ == '__main__':
    a = Attention(128)