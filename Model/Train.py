from Encoder import Encoder
from Decoder import Decoder
from Loss import loss_func, accuracy_fn
from Utils.Constants import *

import tensorflow as tf
import numpy as np
import os
import time
from load_data import load_data
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def readfromjson(path):
    '''
    Read from json

    Parameters:
        path: The path

    Returns:
        The data
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def generate_batch(dataset):
    '''
    Generate a batch

    Returns:
        A batch (batch_pr, batch_prdesc_shift, batch_prdesc)
    '''

    batch_size = 2

    keys = list(dataset.keys())
    N = len(keys)

    for i in range(0, N, batch_size):

        batch_pr = []
        batch_prdesc_shift = []
        batch_prdesc = []

        for j in range(min(batch_size, N-i)):

            key = keys[i+j]
            pr_desc: list = dataset[key]['body']

            batch_pr.append(dataset[key])
            # append start in the beginning
            batch_prdesc_shift.append([0] + pr_desc)
            batch_prdesc.append(pr_desc)

        yield (batch_pr, np.array(batch_prdesc_shift), np.array(batch_prdesc))




@tf.function
def train_step(input_pr, target_prdesc_shift, target_prdesc, encoder: Encoder, decoder: Decoder, optimizer):
    '''
    Train a batch and return loss

    Parameters:
        input_pr: The input pr
            Shape: [dict] * batch_size
        target_prdesc_shift: The shifted target prdesc
            Shape: (batch_size, max_pr_len)
        target_prdesc: The target prdesc
            Shape: (batch_size, max_pr_len)
        encoder: The encoder
        decoder: The decoder
        optimizer: The optimizer
    '''
    
    with tf.GradientTape() as tape:
        # Encode
        h_enc, c_enc = encoder(input_pr)

        # Decode
        logits, _, _ = decoder(target_prdesc_shift, h_enc, c_enc)

        # Calculate loss and accuracy
        loss = loss_func(target_prdesc, logits)
        accuracy = accuracy_fn(target_prdesc, logits)

    # Calculate gradients
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, variables))

    return loss, accuracy

def main_train(encoder:Encoder, decoder:Decoder, dataset, optimizer, epochs, checkpoint, checkpoint_prefix):
    '''
    Train the model

    Parameters:
        encoder: The encoder
        decoder: The decoder
        dataset: The dataset
        optimizer: The optimizer
        epochs: The number of epochs
        checkpoint: The checkpoint
        checkpoint_prefix: The checkpoint prefix
    '''
    losses = []
    accuracies = []

    for epoch in range(epochs):
        # Get start time
        start = time.time()

        # For every batch
        for batch, (batch_pr, batch_prdesc_shift, batch_prdesc) in enumerate(generate_batch(dataset)):
            # Train the batch
            loss, accuracy = train_step(batch_pr, batch_prdesc_shift, batch_prdesc, encoder, decoder, optimizer)
            if batch % 1 == 0:
                losses.append(loss)
                accuracies.append(accuracy)
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, loss.numpy(), accuracy.numpy()))

        # Saving checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        

        print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))

    return losses, accuracies

if __name__ == '__main__':
    # Load dataset
    # dataset = np.load('dataset.npy', allow_pickle=True)
    dataset = load_data(os.path.join('Data', 'sample_dataset_proc.json'))

    # Create encoder and decoder
    encoder = Encoder(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM)
    decoder = Decoder(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM)

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Create checkpoint
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    # Train
    losses, accuracies = main_train(encoder, decoder, dataset, optimizer, 1, checkpoint, checkpoint_prefix)

    # Save losses and accuracies
    np.save('losses.npy', losses)
    np.save('accuracies.npy', accuracies)

    # Save encoder and decoder
    encoder.save_weights('encoder.h5')
    decoder.save_weights('decoder.h5')