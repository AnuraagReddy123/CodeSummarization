import torch
import torch.nn as nn
from Model import Model
from Utils import Constants
from load_data import load_data
import os
from Train import generate_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


if __name__=='__main__':

    dataset = load_data(os.path.join('Data', 'dataset_preproc.json'))

    batch_pr, batch_prdesc_shift, batch_prdesc = generate_batch(dataset, 1)

    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load('Model_Pytorch/model.pt'))

    batch_prdesc = model.predict(batch_pr, Constants.MAX_LEN)



