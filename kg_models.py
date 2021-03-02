import torch
import math
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np


def diversity_regularization(x):
    x = x / F.normalize(x, dim=-1, p=2)
    y = torch.flip(x, [0, 1])

    return torch.cdist(x, y, p=2).mean()


def pairwise_diversity_regularization(x, y):
    x = x / F.normalize(x, dim=-1, p=2)
    y = y / F.normalize(y, dim=-1, p=2)

    return torch.cdist(x, y, p=2).mean()

class SoftplusLoss(nn.Module):

    def __init__(self, adv_temperature = None):
        super(SoftplusLoss, self).__init__()
        self.criterion = nn.Softplus()
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

    def forward(self, p_score, n_score):
        p_score = p_score.flatten()
        n_score = n_score.flatten()
        if self.adv_flag:
            return (self.criterion(-p_score).mean() + (self.get_weights(n_score) * self.criterion(n_score)).sum(dim = -1).mean()) / 2
        else:
            return (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2
            

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()


class DistMult(nn.Module):
    def __init__(self, entity_vocab_size, relation_vocab_size, type_vocab_size, hidden_size, p_norm=1, margin=1,
        embed_type_class=nn.Embedding,
        epsilon = None):

        super(DistMult, self).__init__()

        self.p_norm = p_norm
        self.num_entities = entity_vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.ent_embeddings = embed_type_class(entity_vocab_size, hidden_size)
        self.rel_embeddings = embed_type_class(relation_vocab_size, hidden_size)
        self.type_embeddings = embed_type_class(type_vocab_size, hidden_size)
        self.hidden_size = hidden_size

        self.criterion = SoftplusLoss()

        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.type_embeddings.weight.data)

    def _calc(self, h, t, r):
        score = (h * r) * t

        return torch.sum(score, -1)

    def score(self, h, t, r):
        score = (h * r) * t

        return -torch.sum(score, -1)

    def encode(self, indicies):
        return self.ent_embeddings(indicies)        

    def extract_rel(self, r):
        return self.rel_embeddings(r)



    def calculate_loss(self, pos_pair, neg_pair):
        h, r, t = pos_pair

        neg_h, neg_r, neg_t = neg_pair

        p_score = self._calc(*self.forward(h, r, t))
        n_score = self._calc(*self.forward(neg_h, neg_r, neg_t))

        return self.criterion(p_score, n_score).mean()

    def self_regularization(self):
        return (self.ent_embeddings.weight.norm(p = 3)**3 + \
            self.rel_embeddings.weight.norm(p = 3)**3 + \
            self.type_embeddings.weight.norm(p = 3)**3 )


    def calculate_loss_avg(self, type_triples):
        h, r, t_types, neg_t_types, neg_h = type_triples

        t_types_emb = self.type_embeddings(t_types)

        neg_t_types_emb = self.type_embeddings(neg_t_types)


        p_score = self._calc(self.ent_embeddings(h), t_types_emb.mean(1), self.rel_embeddings(r))
        n_score = self._calc(self.ent_embeddings(neg_h), neg_t_types_emb.mean(1), self.rel_embeddings(r))

        return self.criterion(p_score, n_score),  (
            diversity_regularization( neg_t_types_emb.view(-1, self.hidden_size)) + \
            diversity_regularization( t_types_emb.view(-1, self.hidden_size) )
            ), self.type_regularization( (h, r, t_types) )

    def regularization(self, data):
        batch_h, batch_r, batch_t  = data
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def type_regularization(self, data):
        batch_h, batch_r, batch_t  = data
        h = self.ent_embeddings(batch_h)
        t = self.type_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def forward(self, h, r, t):
        h = self.ent_embeddings(h)
        t = self.ent_embeddings(t)
        r = self.rel_embeddings(r)
        return h, t, r

    def predict(self, h, r, top_k=10):

        rank = []
        for (h_, r_) in zip(h, r):
            h_ = h_.expand(self.entity_vocab_size)
            r_ = r_.expand(self.entity_vocab_size)
            h_ = self.ent_embeddings(h_)
            r_ = self.rel_embeddings(r_)
            t_ = self.ent_embeddings.weight

            p_score = -self._calc( h_, t_, r_ )

            min_values, argsort = torch.sort(p_score, descending=True)
            rank.append(argsort)
        rank = torch.stack(rank)
        # batch size x top_k score index
        return rank


class MarginLoss(nn.Module):

	def __init__(self, adv_temperature = None, margin = 6.0):
		super(MarginLoss, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return torch.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return ((self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin).sum()
		else:
			return ((torch.max(p_score - n_score, -self.margin)).mean() + self.margin).sum()

class TransE(nn.Module):
    def __init__(self, entity_vocab_size, relation_vocab_size, type_vocab_size ,hidden_size, p_norm=1, margin=1, ent_embeddings=None):
        super(TransE, self).__init__()
        self.p_norm = p_norm
        self.num_entities = entity_vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.ent_embeddings = nn.Embedding(entity_vocab_size, hidden_size)
        self.type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.hidden_size = hidden_size

        if ent_embeddings is not None:
            if isinstance(ent_embeddings, nn.Embedding):
                weight_shape = ent_embeddings.weight.data.shape
                self.pre_ent_embeddings = nn.Embedding(*weight_shape)

            self.ent_embeddings = nn.Sequential(self.pre_ent_embeddings, nn.Linear(300, hidden_size))

        self.rel_embeddings = nn.Embedding(relation_vocab_size, hidden_size)
        self.criterion = MarginLoss(margin=margin)
        self.init_weights()

    def init_weights(self):
        if isinstance(self.ent_embeddings, nn.Embedding):
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)

        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.type_embeddings.weight.data)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, self.p_norm, -1)

    def score(self, h, t, r):
        return torch.norm(h + r - t, self.p_norm, -1)


    def encode(self, indicies):
        return self.ent_embeddings(indicies)        

    def extract_rel(self, r):
        return self.rel_embeddings(r)

    def calculate_loss(self, pos_pair, neg_pair):
        h, r, t = pos_pair

        neg_h, neg_r, neg_t = neg_pair

        h_ent, t_ent, r_emb = self.forward(h, r, t)

        p_score = self._calc(h_ent, t_ent, r_emb )
        n_score = self._calc(*self.forward(neg_h, neg_r, neg_t))

        return self.criterion(p_score, n_score)

    def calculate_loss_avg(self, type_triples):
        h, r, t_types, neg_t_types = type_triples

        t_types_emb = self.type_embeddings(t_types)

        neg_t_types_emb = self.type_embeddings(neg_t_types)


        p_score = self._calc(self.ent_embeddings(h), t_types_emb.mean(1), self.rel_embeddings(r) )
        n_score = self._calc(self.ent_embeddings(h), neg_t_types_emb.mean(1), self.rel_embeddings(r) )

        return self.criterion(p_score, n_score),  (
            diversity_regularization( neg_t_types_emb.view(-1, self.hidden_size)) + \
            diversity_regularization( t_types_emb.view(-1, self.hidden_size) )
            ), self.type_regularization( (h, r, t_types) )


    def forward(self, h, r, t):
        h = self.ent_embeddings(h)
        t = self.ent_embeddings(t)
        r = self.rel_embeddings(r)
        return h, t, r

    def predict(self, h, r, top_k=10):

        rank = []
        for (h_, r_) in zip(h, r):
            h_ = h_.expand(self.entity_vocab_size)
            r_ = r_.expand(self.entity_vocab_size)
            h_ = self.ent_embeddings(h_)
            r_ = self.rel_embeddings(r_)
            t_ = self.ent_embeddings.weight
            # v x d, v x d
            p_score = torch.einsum('vd,vd->v', h_+r_, t_)
            # p_score = self._calc(h_, t_, r_)
            min_values, argsort = torch.sort(p_score, descending=True)
            rank.append(argsort)
        rank = torch.stack(rank)
        # batch size x top_k score index
        return rank


    def regularization(self, data):
        batch_h, batch_r, batch_t  = data
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def type_regularization(self, data):
        batch_h, batch_r, batch_t  = data
        h = self.ent_embeddings(batch_h)
        t = self.type_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul


    def self_regularization(self):
        return (self.ent_embeddings.weight.norm(p = 3)**3 + \
            self.rel_embeddings.weight.norm(p = 3)**3 + \
            self.type_embeddings.weight.norm(p = 3)**3 )

