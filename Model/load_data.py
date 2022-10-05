from os import path
import json
from Utils.Structures import Node 


def _build_tree(node, adj):

    for child_id in adj[str(node.id)]['children']:
        
        child = Node(int(child_id))
        node.children.append(child)
        _build_tree(child, adj)
    

def buid_tree(adj):

    if len(adj) == 0:
        return Node(0)
    
    root_id = int(list(adj.keys())[0])
    root = Node(root_id)
    _build_tree(root, adj)

    return root


def load_data(file_path):

    # file_path = path.join('Data', file_name)

    with open(file_path) as f:

        data = json.load(f)
    
    for id in data:

        for commit_sha in data[id]['commits']:

            old_asts = []

            for old_ast in data[id]['commits'][commit_sha]['old_asts']:
                root = buid_tree(old_ast)
                old_asts.append(root)
            
            data[id]['commits'][commit_sha]['old_asts'] = old_asts

            new_asts = []

            for new_ast in data[id]['commits'][commit_sha]['new_asts']:
                root = buid_tree(new_ast)
                new_asts.append(root)
            
            data[id]['commits'][commit_sha]['new_asts'] = new_asts

    return data
        




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


