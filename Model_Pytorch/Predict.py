import torch
import torch.nn as nn
from Model import Model
from Utils import Constants
from load_data import load_data
import os
from Train import generate_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def tensor_to_text(prdesc_tensor, vocab):

    text = ""
    for word_idx in prdesc_tensor:
        text += vocab[int(word_idx)] + " "

    return text

if __name__=='__main__':

    dataset = load_data(os.path.join('Data', 'dataset_preproc.json'))

    batch_pr, batch_prdesc_shift, batch_prdesc = next(generate_batch(dataset, 5))

    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBEDDING_DIM, num_layers=Constants.NUM_LAYERS).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('Model_Pytorch/model_best.pt'))

    pred_batch_prdesc = model.module.predict(batch_pr, Constants.MAX_LEN)

    with open('vocab.txt', 'r') as f:
        vocab = eval(f.read())

    print(len(vocab))

    for i in range(len(batch_pr)):

        print("Ground Truth:")
        print(tensor_to_text(batch_prdesc[i], vocab))
        print()
        print("Prediction:")
        print(tensor_to_text(pred_batch_prdesc[i], vocab))
        print('\n--------------------------------\n')


