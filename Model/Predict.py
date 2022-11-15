import tensorflow as tf
import numpy as np
import os
from Encoder import Encoder
from Decoder import Decoder
from load_data import load_data
from Train import generate_batch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def id2word(id):
    '''
    Map the id to the word

    Parameters:
        id: The id

    Returns:
        The word
    '''
    # Open vocab file
    with open('vocab.txt', 'r') as f:
        vocab = f.read().splitlines()
    
    # Return the word
    return vocab[id]

def predict(input_pr, encoder: Encoder, decoder: Decoder):
    '''
    Predict the description of a pull request

    Parameters:
        input_pr: The input pr
            Shape: [dict] * batch_size
        encoder: The encoder
        decoder: The decoder
    '''

    # Encode the input
    enc_output, enc_hidden = encoder(input_pr)

    # Initialize the decoder
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([0], 0)

    # Initialize the result
    result = []

    for t in range(200):
        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)

        # Get the predicted id
        predicted_id = tf.argmax(predictions[0]).numpy()

        # If the predicted id is 1 (end), stop predicting
        if predicted_id == 2:
            break

        # Append the predicted id to the result
        result.append(predicted_id)

        # The predicted id is the next input to the decoder
        dec_input = tf.expand_dims([predicted_id], 0)

    # Map to the vocabulary
    result = [id2word(i) for i in result]

    return result

if __name__ == "__main__":
    # Load the data
    dataset = load_data(os.path.join('Data', 'dataset_preproc.json'))

    # Create the encoder and decoder
    encoder = Encoder(10000, 256, 1024)
    decoder = Decoder(10000, 256, 1024)

    # Load the weights
    encoder.load_weights('encoder.h5')
    decoder.load_weights('decoder.h5')

    # Generate a batch
    batch_pr, batch_prdesc_shift, batch_prdesc = next(generate_batch(dataset))

    # Predict
    result = predict(batch_pr, encoder, decoder)

    # Print the result
    print(result)