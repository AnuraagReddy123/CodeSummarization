import torch
import torch.nn as nn
from Model import Model
from Utils import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBEDDING_DIM, num_layers=Constants.NUM_LAYERS).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('Model_Pytorch/model_best.pt'))


    print(model.module.encoder.lin_mergeh.weight.data)