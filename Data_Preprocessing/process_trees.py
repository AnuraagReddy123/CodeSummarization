import json
from collections import defaultdict
from os import path
import time
import subprocess

TS_ROOT = path.join('..', 'tree-sitter-codeviews')
INPUT_PATH = path.join(TS_ROOT, 'code_test_files', 'java', 'input.java')
OUTPUT_PATH = path.join(TS_ROOT, 'output_json', 'AST_output.json')


def get_function_index(nodes, function_name):

    n = len(nodes)

    for i, node in enumerate(nodes):

        if node['node_type'] == 'method_declaration':

            if i+1<n and nodes[i+1]['node_type'] == 'identifier' and nodes[i+1]['label'] == function_name:
                return i
            elif i+2<n and nodes[i+2]['node_type'] == 'identifier' and nodes[i+2]['label'] == function_name:
                return i
            elif i+3<n and nodes[i+3]['node_type'] == 'identifier' and nodes[i+3]['label'] == function_name:
                return i
            elif i+4<n and nodes[i+4]['node_type'] == 'identifier' and nodes[i+4]['label'] == function_name:
                return i

    # if the function is not found
    return -1



def dfs(idx, adj_list, vis_nodes_ids:set):

    vis_nodes_ids.add(idx)

    for child_idx in adj_list[idx]:
        dfs(child_idx, adj_list, vis_nodes_ids)


def get_subtree(tree, function_name):

    idx = get_function_index(tree['nodes'], function_name)

    if idx == -1:
        return None

    nodes = tree['nodes']
    edges = tree['links']

    offset = nodes[idx]['id'] - idx

    adj_list = defaultdict(lambda : [])

    for edge in edges:
        adj_list[edge['source']-offset].append(edge['target']-offset)

    vis_nodes_ids = set()
    dfs(idx, adj_list, vis_nodes_ids)

    subtree_nodes = []
    subtree_edges = []

    vis_nodes_ids = list(vis_nodes_ids)
    vis_nodes_ids.sort()

    for i in vis_nodes_ids:
        subtree_nodes.append(nodes[i])
        for j in adj_list[i]:
            subtree_edges.append({'source': i+offset, 'target': j+offset})

    return {
        'nodes': subtree_nodes,
        'links': subtree_edges
    }
    
# def get_custom_ast(ast):

#     '''
#         custom represenation :-
#         {
#             <node_id> : { "label": <label>, "children": [<child_id>, <child_id>, ...] },
#         }
#     '''

#     custom_ast = {}

#     for node in ast['nodes']:

#         node_id = node['id']
#         node_type = node['node_type']
#         node_label = node['label']

#         custom_ast[node_id] = {'label': f'{node_type}_{node_label}', 'children': []}

#     for edge in ast['links']:

#         custom_ast[edge['source']]['children'].append(edge['target'])

#     return custom_ast



default_ast = {
    'nodes': [[-1, -1], [-1, -1]],
    'edges': [[0,1]]
}

def convert_tree(tree):

    if tree is None:
        return default_ast

    # Load the data
    custom_ast = {}
    custom_ast['nodes'] = []

    for node in tree['nodes']:
        custom_ast['nodes'].append([node['node_type'], node['label']])
    
    num = tree['nodes'][0]['id']
    
    custom_ast['edges'] = []

    for edge in tree['links']:
        src, tar = edge['source'] - num, edge['target'] - num
        custom_ast['edges'].append([src, tar])

    return custom_ast


def get_tree(text: str, function_names:list):

    '''
    Returns asts for the given functions
    '''

    open(INPUT_PATH, 'w+').write(text)

    st = time.time()
    subprocess.run('python main.py', shell=True, cwd=TS_ROOT, stdout=subprocess.PIPE)
    ed = time.time()
    print(f'Time taken by tree sitter: {ed - st}')

    full_tree = json.load(open(OUTPUT_PATH))
    final_trees = []
    for func_name in function_names:
        subtree = get_subtree(full_tree, func_name)
        final_tree = convert_tree(subtree)
        final_trees.append(final_tree)


    return final_trees