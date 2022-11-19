from Model import Model
from Loss import loss_fn, accuracy_fn
from Utils.Constants import *

# import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import os
import time
from load_data import load_data
import json
import torch.nn as nn
import torch.optim as optim
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

EPOCHS = 30
BATCH_SIZE = 32

# def readfromjson(path):
#     '''
#     Read from json

#     Parameters:
#         path: The path

#     Returns:
#         The data
#     '''
#     with open(path, 'r') as f:
#         data = json.load(f)
#     return data

def generate_batch(dataset, batch_size):
    '''
    Generate a batch

    Returns:
        A batch (batch_pr, batch_prdesc_shift, batch_prdesc)
    '''

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




# @tf.function
def train_step(input_pr, target_prdesc_shift, target_prdesc, model: Model, optimizer):
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
    logits = model(input_pr, target_prdesc_shift)
    target_prdesc = torch.tensor(target_prdesc, dtype=torch.long, device=device)
    loss = loss_fn(logits, target_prdesc)
    accuracy = accuracy_fn(logits, target_prdesc)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, accuracy

def main_train(model: Model, dataset, optimizer, epochs):
    '''
    Train the model

    Parameters:
        encoder: The encoder
        decoder: The decoder
        dataset: The dataset
        optimizer: The optimizer
        epochs: The number of epochs
    '''
    losses = []
    accuracies = []

    max_accuracy = - math.inf

    for epoch in range(epochs):
        # Get start time
        start = time.time()
        # For every batch
        for batch, (batch_pr, batch_prdesc_shift, batch_prdesc) in enumerate(generate_batch(dataset, BATCH_SIZE)):
            # Train the batch
            loss, accuracy = train_step(batch_pr, batch_prdesc_shift, batch_prdesc, model, optimizer)

            if batch % 1 == 0:
                losses.append(loss)
                accuracies.append(accuracy)
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, loss.item(), accuracy.item()))

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join('Model_Pytorch', 'model.pt'))
                print("Model saved.")
        

        print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))

    return losses, accuracies

if __name__ == '__main__':
    # Load dataset
    dataset = load_data(os.path.join('Data', 'dataset_preproc.json'))

    model = Model(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses, accuracies = main_train(model, dataset, optimizer, epochs=EPOCHS)

    # Save model
    # torch.save(model.state_dict(), os.path.join('Model_Pytorch', 'model.pt'))

    # # Save losses and accuracies
    # np.save(os.path.join('Model_Pytorch', 'losses.npy'), np.array(losses))
    # np.save(os.path.join('Model_Pytorch', 'accuracies.npy'), np.array(accuracies))

    print('Done')