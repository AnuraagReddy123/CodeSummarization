from os import path
import json
from Utils.Structures import Node 
import numpy as np
import copy


# max number of ASTs and Commits 
N_ASTS = 10
N_COMMITS = 10

Empty_commit =  {
    'cm': [1],
    'comments': [1],
    'old_asts': [{}]*N_ASTS,
    'new_asts': [{}]*N_ASTS
}


def adjust_asts(asts: list):

    n = len(asts)

    if n < N_ASTS:
        asts.extend([{}]*(N_ASTS - n))
    elif n > N_ASTS:
        asts = asts[:N_ASTS]

    return asts

def adjust_commits(commits: dict):

    n = len(commits)

    if n < N_COMMITS:
        for i in range(1, N_COMMITS-n+1):
            commits[f'key{i}'] = copy.deepcopy(Empty_commit)
    elif n > N_COMMITS:
        keys = list(commits.keys())
        keys = keys[N_COMMITS:]
        for k in keys:
            del commits[k]

    return commits
    

def _build_tree(node, adj):

    for child_id in adj[str(node.id)]['children']:
        
        child = Node(child_id, adj[str(child_id)]['label'])
        node.children.append(child)
        _build_tree(child, adj)
    

def build_tree(adj):

    if len(adj) == 0:
        # adj = {}
        return Node(0, 1) # 0 -> id, 1 -> _BLANK
    
    root_id = list(adj.keys())[0]
    root = Node(int(root_id), adj[root_id]['label'])
    _build_tree(root, adj)

    return root



'''
Loads the dataset from json file to memory.
Converts lists of numbers to numpy arrays.
Adjust the number of ASTs and Commits.
Builds the tree using the Node data structure.
'''
def load_data(file_path):

    with open(file_path) as f:
        dataset = json.load(f)
    
    for key in dataset:
        dataset[key]['body'] = np.array(dataset[key]['body'] if len(dataset[key]['body']) > 0 else [1])
        dataset[key]['issue_title'] = np.array(dataset[key]['issue_title'] if len(dataset[key]['issue_title']) > 0 else [1])

        commits = dataset[key]['commits']
        commits = adjust_commits(commits)

        for commit_sha in commits:

            commits[commit_sha]['cm'] = np.array(commits[commit_sha]['cm'] if len(commits[commit_sha]['cm']) > 0 else [1])
            commits[commit_sha]['comments'] = np.array(commits[commit_sha]['comments'] if len(commits[commit_sha]['comments']) > 0 else [1])

            old_asts = dataset[key]['commits'][commit_sha]['old_asts']
            if len(old_asts) > 0:
                dataset[key]['commits'][commit_sha]['old_asts'] = [buid_tree(old_ast) for old_ast in old_asts]
            else:
                dataset[key]['commits'][commit_sha]['old_asts'] = [buid_tree({})]

            new_asts = dataset[key]['commits'][commit_sha]['new_asts']
            if len(new_asts) > 0:
                dataset[key]['commits'][commit_sha]['new_asts'] = [buid_tree(new_ast) for new_ast in new_asts]
            else:
                dataset[key]['commits'][commit_sha]['new_asts'] = [buid_tree({})]

    return dataset
        


if __name__ =='__main__':

    data = load_data('../Data/sample_dataset_proc.json')
    root = data['elastic/elasticsearch_37964']['commits']["'df18d6b7d9d2236d1512f7476301ecda15b20401'"]['old_asts'][0]

    def find_height(root):

        if len(root.children) == 0:
            return 1
        
        h_max = 0

        for child in root.children:
            h_max = max(h_max, find_height(child))
        
        return h_max + 1

    print(find_height(root))
