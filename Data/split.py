import json
import sys

train_size = int(sys.argv[1])
valid_size = int(sys.argv[2])
test_size = int(sys.argv[3])

with open('dataset_preproc.json', 'r') as f:
    ds:dict = json.load(f)

keys_all = list(ds.keys())
keys_train = keys_all[:train_size]
keys_valid = keys_all[train_size: train_size+valid_size]
keys_test = keys_all[train_size+valid_size: train_size+valid_size+test_size]

dataset_train = {}
for k in keys_train:
    dataset_train[k] = ds[k]

with open('dataset_train.json', 'w+') as f:
    json.dump(dataset_train, f)

    
dataset_valid = {}
for k in keys_valid:
    dataset_valid[k] = ds[k]

with open('dataset_valid.json', 'w+') as f:
    json.dump(dataset_valid, f)


dataset_test = {}
for k in keys_test:
    dataset_test[k] = ds[k]

with open('dataset_test.json', 'w+') as f:
    json.dump(dataset_test, f)
