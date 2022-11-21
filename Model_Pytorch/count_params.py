import torch
import torch.nn as nn
from Model import Model
from Utils import Constants


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBEDDING_DIM, num_layers=Constants.NUM_LAYERS).to(device)
    model = nn.DataParallel(model)
    print("No. of parameters: ", count_parameters(model))




main()

