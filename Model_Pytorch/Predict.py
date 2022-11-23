import sys
import torch
import torch.nn as nn
from Model import Model
from Utils import Constants
from load_data import load_data
import os
from Train import generate_batch
from Loss import bleu4
from rouge import Rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def tensor_to_text(prdesc_tensor, vocab):

    text = ""
    for word_idx in prdesc_tensor:
        text += vocab[int(word_idx)] + " "

    return text

if __name__=='__main__':

    n_points = int(sys.argv[1])

    dataset = load_data(os.path.join('Data', 'dataset_test.json'))

    batch_pr, batch_prdesc_shift, batch_prdesc = next(generate_batch(dataset, n_points))

    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBEDDING_DIM, num_layers=Constants.NUM_LAYERS).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('Model_Pytorch/model_best.pt'))

    pred_batch_prdesc = model.module.predict(batch_pr, Constants.MAX_LEN)

    with open('Data/vocab.txt', 'r') as f:
        vocab = eval(f.read())

    print(len(vocab))

    r = Rouge()

    for i in range(len(batch_pr)):

        gt = tensor_to_text(batch_prdesc[i], vocab)
        pred = tensor_to_text(pred_batch_prdesc[i], vocab)
        # Take only uptill the END token
        gt1 = gt.split('<END>')[0].strip().split()
        pred1 = pred.split('<END>')[0].strip().split()

        bleu = bleu4(gt1, pred1)
        r_score = r.get_scores(' '.join(pred1), ' '.join(gt1))[0]

        print(f"Ground Truth:\n{gt}\n\nPrediction:\n{pred}\n")
        print(f"Bleu: {bleu}\nRouge-1: {r_score['rouge-1']['f']}\nRouge-2: {r_score['rouge-2']['f']}\nRouge-L: {r_score['rouge-l']['f']}\n\n--------------------\n\n")