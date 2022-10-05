import tensorflow as tf
import numpy as np
import os
from Encoder import Encoder
from Decoder import Decoder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def predict(input_pr, encoder: Encoder, decoder: Decoder):
    '''
    Predict the description of a pull request

    Parameters:
        input_pr: The input pr
            Shape: [dict] * batch_size
        encoder: The encoder
        decoder: The decoder
    '''

    
    return loss, accuracy