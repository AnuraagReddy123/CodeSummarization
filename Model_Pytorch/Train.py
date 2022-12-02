from Model import Model
from Loss import loss_fn, accuracy_fn
from Utils import Constants

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
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def plotter(values, file_name):

    if file_name == 'accuracies':
        plt.ylim([0, 1])
    else:
        plt.ylim([0, 20])

    plt.plot(values)
    plt.savefig(f'Model_Pytorch/{file_name}.png')

    open(f'Model_Pytorch/{file_name}.txt', 'w+').write(str(values))


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
            pr_desc: list = list(dataset[key]['body'])

            batch_pr.append(dataset[key])
            # append start in the beginning
            # prdesc_shift = [0]
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
    # print("in train step")
    model.train()

    logits = model(input_pr, target_prdesc_shift)
    logits = logits[:, :-1]
    target_prdesc = torch.tensor(target_prdesc, dtype=torch.long, device=device)
    loss = loss_fn(logits, target_prdesc)
    accuracy = accuracy_fn(logits, target_prdesc)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, accuracy

def valid_step(input_pr, target_prdesc_shift, target_prdesc, model: Model):

    model.eval()

    with torch.no_grad():

        logits = model(input_pr, target_prdesc_shift)
        logits = logits[:, :-1]
        target_prdesc = torch.tensor(target_prdesc, dtype=torch.long, device=device)
        loss = loss_fn(logits, target_prdesc)
        accuracy = accuracy_fn(logits, target_prdesc)

        return loss, accuracy


def main_train(model: Model, dataset_train, dataset_valid, optimizer, epochs):
    '''
    Train the model

    Parameters:
        encoder: The encoder
        decoder: The decoder
        dataset: The dataset
        optimizer: The optimizer
        epochs: The number of epochs
    '''
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    max_accuracy_train = - math.inf
    max_accuracy_valid = - math.inf

    for epoch in range(epochs):
        # print(f"epoch: {epoch+1}")
        # Get start time
        start = time.time()
        # For every batch
        for batch, (batch_pr, batch_prdesc_shift, batch_prdesc) in enumerate(generate_batch(dataset_train, Constants.BATCH_SIZE)):

            # if batch > 0:
            #     continue
            # print(f"batch: {batch}")

            # Train the batch
            train_loss, train_accuracy = train_step(batch_pr, batch_prdesc_shift, batch_prdesc, model, optimizer)
            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy.item())
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.item(), train_accuracy.item()))

            if train_accuracy.item() > max_accuracy_train:
                max_accuracy_train = train_accuracy.item()
                torch.save(model.state_dict(), os.path.join('Model_Pytorch', 'model_best_train.pt'))
                print("Model Train saved.")

        # validate the model
        for batch, (batch_pr, batch_prdesc_shift, batch_prdesc) in enumerate(generate_batch(dataset_valid, len(dataset_valid))):

            valid_loss, valid_accuracy = valid_step(batch_pr, batch_prdesc_shift, batch_prdesc, model)
            valid_losses.append(valid_loss.item())
            valid_accuracies.append(valid_accuracy.item())
            print('Epoch {} Validation, Loss {:.4f} Accuracy {}'.format(epoch + 1, valid_loss.item(), valid_accuracy.item()))

            print(valid_accuracy.item(), max_accuracy_valid, valid_accuracy.item() > max_accuracy_valid)

            if valid_accuracy.item() > max_accuracy_valid:
                max_accuracy_valid = valid_accuracy.item()
                torch.save(model.state_dict(), os.path.join('Model_Pytorch', 'model_best_valid.pt'))
                print("Model Valid saved.")


        print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))

    return train_losses, train_accuracies

if __name__ == '__main__':
    # Load dataset
    dataset_train = load_data(os.path.join('Data', 'dataset_train.json'))
    dataset_valid = load_data(os.path.join('Data', 'dataset_valid.json'))

    print("loaded data.")

    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBEDDING_DIM, num_layers=Constants.NUM_LAYERS).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("created model.")

    losses, accuracies = main_train(model, dataset_train, dataset_valid, optimizer, epochs=Constants.EPOCHS)

    plotter(losses, 'losses')
    plotter(accuracies, 'accuracies')

    # Save model
    torch.save(model.state_dict(), os.path.join('Model_Pytorch', 'model_final.pt'))

    print('Done')
