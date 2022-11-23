import os
from os import path
import json
import time
from github import Github
import requests
import shlex
import subprocess
import whatthepatch
from dotenv import load_dotenv
from process_trees import get_tree

load_dotenv()
github = Github(os.environ['TOKEN'])


MAX_ASTS = 1

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
    proc = subprocess.run(f'git show {file_sha}', shell=True, cwd=repo_path, stdout=subprocess.PIPE, text=True)
    cur_text = proc.stdout

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






if __name__=='__main__':

    st_g = time.time()

    if not path.isdir(path.join('repos')):
        os.makedirs(path.join('repos'))

    with open(path.join('..', 'Data', 'dataset.json')) as f:
        dataset = json.load(f)
    
    user, repo = [None]*2

    i = 1

    for d_key in dataset:

        print(f'\n--- datapoint {i} -------------------\n')
        i += 1

        username, repo_name, pull_number = parse_key(d_key)

        user, repo, pull_req = get_obj(username, repo_name, pull_number, user, repo)

        # -------------- add issue title --------------------

        try:
            issue_res = requests.get(pull_req.issue_url)
            dataset[d_key]['issue_title'] = issue_res.json()['title']
        except:
            print("No issue associated.")
            dataset[d_key]['issue_title'] = ''

        print("issue title check.")

        # ---------------- add ASTs ---------------------------------------
        clone_repo(username, repo_name)
        repo_path = path.join('repos', username, repo_name)

        for commit in pull_req.get_commits():

            dataset[d_key]['commits'][f"'{commit.sha}'"]['cur_asts'] = []
            dataset[d_key]['commits'][f"'{commit.sha}'"]['old_asts'] = []

            print(f'COMMIT {commit.sha}: {len(commit.files)} files.')

            for file in commit.files:

                if len(dataset[d_key]['commits'][f"'{commit.sha}'"]['cur_asts']) >= MAX_ASTS:
                    break

                # Considering only the changes in JAVA files.
                if not file.filename.endswith('.java'):
                    continue

                try:
                    cur_text = get_cur_version(repo_path, file.sha)
                    old_text = get_prev_version(cur_text, file.patch)
                except:
                    print("Continuing....")
                    continue

                func_names = get_entity_names(file.patch)

                cur_asts = get_tree(cur_text, func_names)
                old_asts = get_tree(old_text, func_names)

                dataset[d_key]['commits'][f"'{commit.sha}'"]['cur_asts'].extend(cur_asts)
                dataset[d_key]['commits'][f"'{commit.sha}'"]['old_asts'].extend(old_asts)
        
    with open(path.join('..', 'Data', 'dataset_aug.json'), 'w+') as f:
        json.dump(dataset, f)
    
    ed_g = time.time()
    # TIME
    print(f'Time taken overall : {ed_g - st_g}')
            

