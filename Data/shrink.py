import json


with open('dataset_all.json', 'r') as f:
    dataset_all:dict = json.load(f)


dataset_sh = {}

keys_all = list(dataset_all.keys())
keys_sh = keys_all[:1000]

for k in keys_sh:
    dataset_sh[k] = dataset_all[k]

with open('dataset.json', 'w+') as f:
    json.dump(dataset_sh, f)
