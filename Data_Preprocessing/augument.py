import os
from os import path
import json
import time
from github import Github
import pygraphviz
import requests
import shlex
import subprocess
import whatthepatch
from dotenv import load_dotenv

load_dotenv()

github = Github(os.environ['TOKEN'])

def parse_key(d_key):

    '''
        Parses the dataset key which is in the form
        <user>/<repo>_<pull_number>
    '''

    username = d_key.split('/')[0]
    repo_name = d_key.split('/')[1].split('_')[0]
    pull_number = int(d_key.split('/')[1].split('_')[1])

    return username, repo_name, pull_number

def get_obj(username, repo_name, pull_number, user, repo):

    '''
        Calls the Github api to fetch the user, repo, and pull objects
    '''

    if not user or not user.name == username:
        user = github.get_user(username)
        repo = user.get_repo(repo_name)
    elif not repo or not repo.name == repo_name:
        repo = user.get_repo(repo_name)

    pull_req = repo.get_pull(pull_number)

    return user, repo, pull_req


def clone_repo(username, repo_name):

        '''
            Clones the repo if it doesn't exist already
        '''

        repo_path = path.join('repos', username, repo_name)
        if not path.isdir(repo_path):
            os.makedirs(repo_path)
            command = shlex.split(f'git clone https://github.com/{username}/{repo_name}.git {repo_path}')
            subprocess.run(command)

        print("repo check.")

def get_entity_names(patch: str):

    '''
        Parses the git diff patch of a file and returns the 
        entities (method, class, etc.) names where the changes 
        occurred. These entity names are used to identify the 
        ASTs.
    '''

    lines = patch.split('\n')
    lines = list(filter(lambda x: x[0]=='@', lines))
    entity_names = [x.split('@@')[-1].split('(')[0].split(' ')[-1] for x in lines]
    return entity_names

def get_cur_version(repo_path, file_sha):
    '''
        Retrieves the version of the file after modifcation 
        using the file sha
    '''
    pipe = os.popen(f'cd {repo_path} && git show {file_sha}')
    cur_text = pipe.read()
    pipe.close()

    return cur_text

def get_prev_version(cur_text, file_patch):

    '''
        Applies the patch backwards to retrieve the version of the
        file before modification.
    '''

    diff_obj = [x for x in whatthepatch.parse_patch(file_patch)][0]
    old_text = whatthepatch.apply_diff(diff_obj, cur_text, reverse=True)
    old_text = '\n'.join(old_text)

    return old_text


def get_custom_asts(g_asts):

    '''
        Converts the ASTs from the graphviz representation 
        to custom json representation.

        custom represenation :-
        {
            <node_id> : { "label": <label>, "children": [<child_id>, <child_id>, ...] },
        }
    '''
    asts = []

    for g in g_asts:
        if g is None:
            asts.append({})
            continue
        tree = {}
        for node in g.nodes():
            label = node.attr['label'].split('(')[1].split(')')[0].split(',')[0]
            children = [int(x) for x in g.out_neighbors(node)]
            tree[node.name] = {'label': label, 'children': children}

        asts.append(tree)

    return asts



if __name__=='__main__':

    if not path.isdir('repos'):
        os.makedirs('repos')

    with open(path.join('..', 'Data', 'dataset.json')) as f:
        dataset = json.load(f)
    
    user, repo = [None]*2

    for d_key in dataset:

        username, repo_name, pull_number = parse_key(d_key)

        st = time.time()
        user, repo, pull_req = get_obj(username, repo_name, pull_number, user, repo)
        ed = time.time()
        # TIME
        print(f'Time taken for get obj using github api: {ed - st}')

        try:
            issue_res = requests.get(pull_req.issue_url)
            dataset[d_key]['issue_title'] = issue_res.json()['title']
        except:
            print("No issue associated.")
            dataset[d_key]['issue_title'] = ''

        print("issue title check.")

        # ---------------- ASTs ---------------------------------------

        clone_repo(username, repo_name)
        repo_path = path.join('repos', username, repo_name)

        for commit in pull_req.get_commits():

            dataset[d_key]['commits'][f"'{commit.sha}'"]['cur_asts'] = []
            dataset[d_key]['commits'][f"'{commit.sha}'"]['old_asts'] = []

            for file in commit.files:
                # Considering only the changes in JAVA files.
                if file.filename.endswith('.java'):

                    st = time.time()
                    cur_text = get_cur_version(repo_path, file.sha)
                    ed = time.time()
                    # TIME
                    print(f'Time taken for get cur ver : {ed - st}')
                    old_text = get_prev_version(cur_text, file.patch)

                    print(file.patch)

                    entity_names = get_entity_names(file.patch)

                    os.makedirs('temp')

                    with open('./temp/cur.java', 'w+') as f:
                        f.write(cur_text)
                    with open('./temp/old.java', 'w+') as f:
                        f.write(old_text)
                    
                    
                    st = time.time()
                    os.popen(f'cd temp && joern-parse cur.java && joern-export --repr ast --out cur_ast').close()
                    os.popen(f'cd temp && joern-parse old.java && joern-export --repr ast --out old_ast').close()
                    ed = time.time()
                    # TIME
                    print(f'Time taken for Joern parsing : {ed - st}')

                    cur_asts_dict = {}
                    for filename in os.listdir(path.join('temp', 'cur_ast')):
                        g = pygraphviz.AGraph()
                        g.read(path.join('temp', 'cur_ast', filename))
                        cur_asts_dict[g.name] = g

                    old_asts_dict = {}
                    for filename in os.listdir(path.join('temp', 'old_ast')):
                        g = pygraphviz.AGraph()
                        g.read(path.join('temp', 'old_ast', filename))
                        old_asts_dict[g.name] = g
                        
                    g_names = sorted(set(old_asts_dict.keys()).union(set(cur_asts_dict.keys())))

                    cur_asts_list = [cur_asts_dict.get(g_name, None) for g_name in g_names if g_name in entity_names]
                    old_asts_list = [old_asts_dict.get(g_name, None) for g_name in g_names if g_name in entity_names]

                    st = time.time()
                    cur_asts_list = get_custom_asts(cur_asts_list)
                    old_asts_list = get_custom_asts(old_asts_list)
                    ed = time.time()
                    # TIME
                    print(f'Time taken for custom ASTs building: {ed - st}')

                    dataset[d_key]['commits'][f"'{commit.sha}'"]['cur_asts'].extend(cur_asts_list)
                    dataset[d_key]['commits'][f"'{commit.sha}'"]['old_asts'].extend(old_asts_list)

                    exit(0)
                    
                    os.popen('rm -rf temp').close()

        
    
    # with open(path.join('..', 'Data', 'dataset_aug.json'), 'w+') as f:
        # json.dump(dataset, f)
            

