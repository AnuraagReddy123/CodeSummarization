from os import path
import json
from Utils.Structures import Node 


def _build_tree(node, adj):

    for child_id in adj[str(node.id)]['children']:
        
        child = Node(child_id, adj[str(child_id)]['label'])
        node.children.append(child)
        _build_tree(child, adj)
    

def buid_tree(adj):

    if len(adj) == 0:
        return Node(0, 1) # 0 -> id, 1 -> _UNK
    
    root_id = list(adj.keys())[0]
    root = Node(int(root_id), adj[root_id]['label'])
    _build_tree(root, adj)

    return root


# Loads the dataset, converts the json to Node structure
def load_data(file_path):

    with open(file_path) as f:
        dataset = json.load(f)
    
    for key in dataset:
        for commit_sha in dataset[key]['commits']:

            old_asts = dataset[key]['commits'][commit_sha]['old_asts']
            dataset[key]['commits'][commit_sha]['old_asts'] = [buid_tree(old_ast) for old_ast in old_asts]

            new_asts = dataset[key]['commits'][commit_sha]['new_asts']
            dataset[key]['commits'][commit_sha]['new_asts'] = [buid_tree(new_ast) for new_ast in new_asts]

    return dataset



        
'''
data = load_data('sample_dataset_proc.json')

root = data['elastic/elasticsearch_37964']['commits']["'df18d6b7d9d2236d1512f7476301ecda15b20401'"]['old_asts'][0]

def find_height(root):

    if len(root.children) == 0:
        return 1
    
    h_max = 0

    for child in root.children:
        h_max = max(h_max, find_height(child))
    
    return h_max + 1


print(find_height(root))
'''


