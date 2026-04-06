import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BPR(nn.Module):
    def __init__(self, data_config, args):
        super(BPR, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size
        self.regs = args.regs
        self.decay = self.regs


        self.user_embeddings = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.emb_dim)


        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def forward(self, user_indices, item_indices):

        user_emb = self.user_embeddings(user_indices)
        item_emb = self.item_embeddings(item_indices)
        pred = torch.mul(user_emb, item_emb).sum(dim=1)
        return pred

    def predict(self, users, items):

        self.eval()
        with torch.no_grad():
            users = torch.from_numpy(users).long().cuda()
            items = torch.from_numpy(items).long().cuda()
            pred = self.forward(users, items)
        return pred.cpu().numpy()

    def batch_rating(self, user_indices, candidate_items):

        self.eval()
        with torch.no_grad():
            user_emb = self.user_embeddings(user_indices)
            item_emb = self.item_embeddings(candidate_items)
            rating = torch.matmul(user_emb, item_emb.t())
        return rating.cpu().numpy()

    def train_one_batch(self, users, pos_items, neg_items, optimizer):

        self.train()
        optimizer.zero_grad()


        user_emb = self.user_embeddings(users)
        pos_item_emb = self.item_embeddings(pos_items)
        neg_item_emb = self.item_embeddings(neg_items)

        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)


        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()


        reg_loss = self.decay * (
                torch.norm(user_emb, p=2, dim=1).mean() +
                torch.norm(pos_item_emb, p=2, dim=1).mean() +
                torch.norm(neg_item_emb, p=2, dim=1).mean()
        )


        total_loss = bpr_loss + reg_loss
        total_loss.backward()
        optimizer.step()

        return bpr_loss.item(), reg_loss.item(), total_loss.item()

    def get_embeddings(self):
        self.eval()
        with torch.no_grad():
            user_emb = self.user_embeddings.weight.data.cpu().numpy()
            item_emb = self.item_embeddings.weight.data.cpu().numpy()
        return user_emb, item_emb
