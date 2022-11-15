import torch 
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim, emb_dim):
        super(Encoder, self).__init__()

        self.emb_commit_msgs = nn.Embedding(vocab_size, emb_dim)
        self.emb_src_comments = nn.Embedding(vocab_size, emb_dim)
        self.emb_issue_titles = nn.Embedding(vocab_size, emb_dim)

        self.enc_commit_msgs = nn.LSTM(emb_dim, hidden_dim, 1, batch_first=True)
        self.enc_src_comments = nn.LSTM(emb_dim, hidden_dim, 1, batch_first=True)
        self.emb_issue_titles = nn.LSTM(emb_dim, hidden_dim, 1, batch_first=True)

    def initialize_hidden_state(self):
        return torch.zeros((1, 1, self.hidden_dim)), torch.zeros((1, 1, self.hidden_dim))

    def forward(self):
        pass

    def encoode (self, pr):
        commits = pr['commits']
        
        enc_commits = []
        h_commits = []
        c_commits = []

        for commit in commits:
            inp_sc = commit['comments']
            inp_commit = commit['cm']

            # Increase dim
            inp_sc = inp_sc.unsqueeze(0)
            inp_commit = inp_commit.unsqueeze(0)

            # Embedding
            emb_src_comments = self.emb_src_comments(inp_sc) # (1, seq_len, emb_dim)
            emb_commit_msgs = self.emb_commit_msgs(inp_commit) # (1, seq_len, emb_dim)

            # Encoding
            h0, c0 = self.initialize_hidden_state()
            enc_src_comments, (h_src_comments, c_src_comments) = self.enc_src_comments(emb_src_comments, (h0, c0)) # (1, seq_len, hidden_dim), (1, 1, hidden_dim), (1, 1, hidden_dim)

            h0, c0 = self.initialize_hidden_state()
            enc_commit_msgs, (h_commit_msgs, c_commit_msgs) = self.enc_commit_msgs(emb_commit_msgs, (h0, c0)) # (1, seq_len, hidden_dim), (1, 1, hidden_dim), (1, 1, hidden_dim)

            # Concatenate
            h_commit = torch.cat((h_src_comments, h_commit_msgs), dim=2) # (1, 1, 2*hidden_dim)
            c_commit = torch.cat((c_src_comments, c_commit_msgs), dim=2) # (1, 1, 2*hidden_dim)

            h_commits.append(h_commit)
            c_commits.append(c_commit)
        
        