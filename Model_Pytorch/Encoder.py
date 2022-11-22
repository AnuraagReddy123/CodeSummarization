import torch 
import torch.nn as nn
from Utils import Constants
import treelstm

MAX_ASTS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Encoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim, embed_dim, num_layers):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.emb_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.emb_commit_msgs = nn.Embedding(vocab_size, embed_dim)
        self.emb_src_comments = nn.Embedding(vocab_size, embed_dim)
        self.emb_issue_titles = nn.Embedding(vocab_size, embed_dim)

        self.enc_commit_msgs = nn.LSTM(embed_dim, hidden_dim,  num_layers=num_layers, batch_first=True, dropout=0.1)
        self.enc_src_comments = nn.LSTM(embed_dim, hidden_dim,  num_layers=num_layers, batch_first=True, dropout=0.1)
        self.enc_issue_titles = nn.LSTM(embed_dim, hidden_dim,  num_layers=num_layers, batch_first=True, dropout=0.1)

        self.tree_lstm = treelstm.TreeLSTM(2, hidden_dim)
        self.lin_astdiffh = nn.Linear(2*hidden_dim, hidden_dim)
        self.lin_astdiffc = nn.Linear(2*hidden_dim, hidden_dim)

        self.lin_astmergeh = nn.Linear(hidden_dim, 1)
        self.lin_astmergec = nn.Linear(hidden_dim, 1)

        self.lin_mergeh = nn.Linear(2*hidden_dim+MAX_ASTS, 1)
        self.lin_mergec = nn.Linear(2*hidden_dim+MAX_ASTS, 1)

        self.lin_finmergeh = nn.Linear(Constants.MAX_COMMITS+hidden_dim, hidden_dim)
        self.lin_finmergec = nn.Linear(Constants.MAX_COMMITS+hidden_dim, hidden_dim)

    def initialize_hidden_state(self):
        return torch.zeros((self.num_layers, 1, self.hidden_dim)).to(device), torch.zeros((self.num_layers, 1, self.hidden_dim)).to(device)

    def forward(self, batch_pr):

        batch_h = []
        batch_c = []

        for pr in batch_pr:
            h, c = self.encode(pr)
            batch_h.append(h)
            batch_c.append(c)
        
        batch_h = torch.cat(batch_h, dim=1) # (num_layers, batch_size, hidden_dim)
        batch_c = torch.cat(batch_c, dim=1) # (num_layers, batch_size, hidden_dim)

        return batch_h, batch_c

    def encode (self, pr):
        commits = pr['commits']
        
        enc_commits = []
        h_commits = []
        c_commits = []

        for commit in commits.values():
            inp_sc = commit['comments']
            inp_commit = commit['cm']

            # convert to tensor
            inp_sc = torch.tensor(inp_sc).to(device)
            inp_commit = torch.tensor(inp_commit).to(device)

            # Increase dim
            inp_sc = inp_sc.unsqueeze(0)
            inp_commit = inp_commit.unsqueeze(0)

            # Embedding
            emb_src_comments = self.emb_src_comments(inp_sc) # (1, seq_len, emb_dim)
            emb_commit_msgs = self.emb_commit_msgs(inp_commit) # (1, seq_len, emb_dim)

            # Encoding
            h0, c0 = self.initialize_hidden_state()
            enc_src_comments, (h_src_comments, c_src_comments) = self.enc_src_comments(emb_src_comments, (h0, c0)) # (batch_size=1, seq_len, hidden_dim), (num_layers, 1, hidden_dim), (num_layers, 1, hidden_dim)

            h0, c0 = self.initialize_hidden_state()
            enc_commit_msgs, (h_commit_msgs, c_commit_msgs) = self.enc_commit_msgs(emb_commit_msgs, (h0, c0)) # (1, seq_len, hidden_dim), (num_layers, 1, hidden_dim), (num_layers, 1, hidden_dim)

            old_asts = commit['old_asts']
            cur_asts = commit['cur_asts']

            h_asts = []
            c_asts = []
            for old, cur in zip(old_asts, cur_asts):
                h_old, c_old = self.tree_lstm(old['features'], old['node_order'], old['adjacency_list'], old['edge_order'])
                h_cur, c_cur = self.tree_lstm(cur['features'], cur['node_order'], cur['adjacency_list'], cur['edge_order'])
                h_old, c_old = h_old[0], c_old[0]
                h_cur, c_cur = h_cur[0], c_cur[0]
                h_ast = self.lin_astdiffh(torch.cat((h_old, h_cur)).unsqueeze(0)) # (1, hidden_dim)
                c_ast = self.lin_astdiffc(torch.cat((c_old, c_cur)).unsqueeze(0)) # (1, hidden_dim)

                h_asts.append(h_ast.squeeze())
                c_asts.append(c_ast.squeeze())
            
            h_asts = torch.stack(h_asts, dim=0) # (num_of_trees, hidden_dim)
            c_asts = torch.stack(c_asts, dim=0) # (num_of_trees, hidden_dim)

            h_asts = torch.t(self.lin_astmergeh(h_asts)) # (1, num_of_trees)
            c_asts = torch.t(self.lin_astmergec(c_asts)) # (1, num_of_trees)

            h_asts = torch.stack([h_asts]*self.num_layers) # (num_layers, 1, num_of_trees)
            c_asts = torch.stack([c_asts]*self.num_layers) # (num_layers, 1, num_of_trees)

            # Concatenate
            h_commit = torch.cat((h_src_comments, h_commit_msgs, h_asts), dim=2) # (num_layers, batch_size=1, 2*hidden_dim+num_of_trees)
            c_commit = torch.cat((c_src_comments, c_commit_msgs, h_asts), dim=2) # (num_layers, batch_size=1, 2*hidden_dim+num_of_trees)

            h_commits.append(h_commit)
            c_commits.append(c_commit)
        
        # Make tensor
        h_commits = torch.cat(h_commits, dim=1) # (num_layers, num_commits, 2*hidden_dim+num_of_trees)
        c_commits = torch.cat(c_commits, dim=1) # (num_layers, num_commits, 2*hidden_dim+num_of_trees)

        # Merge all commits
        h_commits = self.lin_mergeh(h_commits) # (num_layers, num_commits, 1)
        c_commits = self.lin_mergec(c_commits) # (num_layers, num_commits, 1)

        # Transpose
        h_commits = h_commits.transpose(1, 2) # (num_layers, 1, num_commits)
        c_commits = c_commits.transpose(1, 2) # (num_layers, 1, num_commits)


        # Encode the issue
        inp_issue = pr['issue_title']
        inp_issue = torch.tensor(inp_issue).to(device)
        inp_issue = inp_issue.unsqueeze(0)

        emb_issue_titles = self.emb_issue_titles(inp_issue) # (1, seq_len, emb_dim)
        h0, c0 = self.initialize_hidden_state()
        enc_issue_titles, (h_issue_titles, c_issue_titles) = self.enc_issue_titles(emb_issue_titles, (h0, c0)) # (1, seq_len, hidden_dim), (num_layers, 1, hidden_dim), (num_layers, 1, hidden_dim)

        # Concatenate
        h = torch.cat((h_commits, h_issue_titles), dim=2) # (num_layers, 1, num_commits+hidden_dim)
        c = torch.cat((c_commits, c_issue_titles), dim=2) # (num_layers, 1, num_commits+hidden_dim)

        # Merge
        h = self.lin_finmergeh(h) # (num_layers, 1, hidden_dim)
        c = self.lin_finmergec(c) # (num_layers, 1, hidden_dim)

        return h, c


if __name__ == '__main__':

    vocab_size = 100
    hidden_dim = 10
    emb_dim = 5

    batch_size = 2
    num_commits = 10
    max_seq_len = 100

    # batch_src_comments = torch.randint(0, vocab_size, (batch_size, num_commits, max_seq_len))

    # encoder = Encoder(vocab_size, hidden_dim, emb_dim)
    # enc_src_comments, h, c = encoder(batch_src_comments)

    # print(enc_src_comments.shape)

    # exit(0)

    batch_pr = []
    for i in range(batch_size):
        pr = {}
        pr['issue'] = torch.randint(0, vocab_size, (max_seq_len,))
        pr['commits'] = []
        for j in range(num_commits):
            commit = {}
            commit['cm'] = torch.randint(0, vocab_size, (max_seq_len,))
            commit['comments'] = torch.randint(0, vocab_size, (max_seq_len,))
            pr['commits'].append(commit)
        batch_pr.append(pr)

    encoder = Encoder(vocab_size, hidden_dim, emb_dim)
    h, c = encoder(batch_pr)
    print(h.shape) # (1, batch_size, hidden_dim)
    print(c.shape) # (1, batch_size, hidden_dim)
    print(h)
    print(c)