import torch
import torch.nn as nn

def loss_fn (logits, target_prdesc):
    '''
    Calculate the masked loss

    Parameters:
        logits: The logits
            Shape: (batch_size, max_pr_len, vocab_size)
        target_prdesc: The target prdesc
            Shape: (batch_size, max_pr_len)
    '''
    # Mask the loss
    mask = (target_prdesc != 1).float() # The padding value is 1
    print("Mask: ", mask)
    # Transpose the logits
    logits = logits.transpose(1, 2)
    loss = nn.CrossEntropyLoss(reduction='none')(logits, target_prdesc)
    loss = loss * mask
    loss = loss.sum() / mask.sum()

    return loss

def accuracy_fn (logits, target_prdesc):
    '''
    Calculate the masked accuracy

    Parameters:
        logits: The logits
            Shape: (batch_size, max_pr_len, vocab_size)
        target_prdesc: The target prdesc
            Shape: (batch_size, max_pr_len)
    '''
    # Mask the accuracy
    mask = (target_prdesc != 1).float() # The padding value is 1
    pred = torch.argmax(logits, dim=-1)
    correct = (pred == target_prdesc).float()
    correct = correct * mask
    accuracy = correct.sum() / mask.sum()

    print("Compare: ")
    print(pred)
    print(target_prdesc)
    print("Correct: ", correct.sum(), " Mask: ", mask.sum())

    return accuracy 
    