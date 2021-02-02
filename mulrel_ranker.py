import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

from DCA.local_ctx_att_ranker import LocalCtxAttRanker
from torch.distributions import Categorical
import copy

np.set_printoptions(threshold=20)

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




class MulRelRanker(LocalCtxAttRanker):
    """
    multi-relational global model with context token attention, using loopy belief propagation
    """

    def __init__(self, config):

        print('--- create MulRelRanker model ---')
        super(MulRelRanker, self).__init__(config)
        self.dr = config['dr']
        self.gamma = config['gamma']
        self.tok_top_n4ment = config['tok_top_n4ment']
        self.tok_top_n4ent = config['tok_top_n4ent']
        self.tok_top_n4word = config['tok_top_n4word']
        self.tok_top_n4inlink = config['tok_top_n4inlink']
        self.order_learning = config['order_learning']
        self.dca_method = config['dca_method']

        self.ent_unk_id = config['entity_voca'].unk_id
        self.word_unk_id = config['word_voca'].unk_id
        self.ent_inlinks = config['entity_inlinks']

        # self.oracle = config.get('oracle', False)
        self.use_local = config.get('use_local', False)
        self.use_local_only = config.get('use_local_only', False)
        self.freeze_local = config.get('freeze_local', False)

        self.entity2entity_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))
        self.entity2entity_score_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))

        self.knowledge2entity_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))
        self.knowledge2entity_score_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))

        self.ment2ment_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))
        self.ment2ment_score_mat_diag = torch.nn.Parameter(torch.ones(self.emb_dims))

        self.cnn = torch.nn.Conv1d(self.emb_dims, 64, kernel_size=3)

        self.saved_log_probs = []
        self.rewards = []
        self.actions = []

        self.order_saved_log_probs = []
        self.decision_order = []
        self.targets = []
        self.record = False
        if self.freeze_local:
            self.att_mat_diag.requires_grad = False
            self.tok_score_mat_diag.requires_grad = False

        self.ment2ment_mat_diag.requires_grad = False
        self.ment2ment_score_mat_diag.requires_grad = False
        self.param_copy_switch = True

        # Typing feature
        self.type_emb = torch.nn.Parameter(torch.randn([4, 5]))

        # type of embedding
        self.type_embeddings = nn.Embedding( config['type_vocab_size'], self.emb_dims)
        self.rel_embeddings = nn.Embedding( config['rel_vocab_size'], self.emb_dims)
        self.criterion = SoftplusLoss()


        self.score_combine = torch.nn.Sequential(
                torch.nn.Linear(5, self.hid_dims),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dr),
                torch.nn.Linear(self.hid_dims, 1))

        # self.score_combine = torch.nn.Sequential(
        #         torch.nn.Linear(5, self.hid_dims),
        #         torch.nn.ReLU(),
        #         torch.nn.Dropout(p=self.dr),
        #         torch.nn.Linear(self.hid_dims, 1))

        self.flag = 0
        # print('---------------- model config -----------------')
        # for k, v in self.__dict__.items():
        #     if not hasattr(v, '__dict__'):
        #         print(k, v)
        # print('-----------------------------------------------')

    # def switch_order_learning(self, option):
    #     if option == 1:
    #         self.ment2ment_mat_diag.requires_grad = True
    #         self.ment2ment_score_mat_diag.requires_grad = True
    #
    #         self.entity2entity_mat_diag.requires_grad = False
    #         self.entity2entity_score_mat_diag.requires_grad = False
    #         self.knowledge2entity_mat_diag.requires_grad = False
    #         self.knowledge2entity_score_mat_diag.requires_grad = False
    #     elif option == 0:
    #         self.ment2ment_mat_diag.requires_grad = False
    #         self.ment2ment_score_mat_diag.requires_grad = False
    #
    #         self.entity2entity_mat_diag.requires_grad = True
    #         self.entity2entity_score_mat_diag.requires_grad = True
    #         self.knowledge2entity_mat_diag.requires_grad = True
    #         self.knowledge2entity_score_mat_diag.requires_grad = True

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m, mtype, etype, ment_ids, ment_mask, desc_ids, desc_mask, gold=None,
                method="SL", isTrain=True, isDynamic=0, isOrderLearning=False, isOrderFixed=False, isSort='topic', basemodel='deeped'):

        n_ments, n_cands = entity_ids.size()

        # cnn
        desc_len = desc_ids.size(-1)

        context_len = token_ids.size(-1)

        desc_ids = desc_ids.view(n_ments*n_cands, -1)
        desc_mask = desc_mask.view(n_ments*n_cands, 1, -1)

        context_emb = self.word_embeddings(token_ids)
        desc_emb = self.word_embeddings(desc_ids)

        context_cnn = F.max_pool1d(self.cnn(context_emb.permute(0, 2, 1)), context_len-2).permute(0, 2, 1)

        desc_cnn = F.max_pool1d(self.cnn(desc_emb.permute(0, 2, 1))-(1-desc_mask.float())*1e10, desc_len-2).view(n_ments, n_cands, -1)

        sim = torch.sum(context_cnn*desc_cnn,-1) / torch.sqrt(torch.sum(context_cnn*context_cnn, -1)) / torch.sqrt(torch.sum(desc_cnn*desc_cnn, -1))

        # if not self.oracle:
        #     gold = None

        # Typing feature
        self.mt_emb = torch.matmul(mtype, self.type_emb).view(n_ments, 1, -1)
        self.et_emb = torch.matmul(etype.view(-1, 4), self.type_emb).view(n_ments, n_cands, -1)
        tm = torch.sum(self.mt_emb*self.et_emb, -1, True)

        if self.use_local:
            local_ent_scores = super(MulRelRanker, self).forward(token_ids, tok_mask, entity_ids, entity_mask,
                                                                 p_e_m=None)
        else:
            local_ent_scores = Variable(torch.zeros(n_ments, n_cands).cuda(), requires_grad=False)

        if self.use_local_only:
            # Typing feature
            inputs = torch.cat([sim.view(n_ments * n_cands, -1) if not basemodel == 'deeped' else local_ent_scores.view(n_ments * n_cands, -1),
                                torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1), tm.view(n_ments * n_cands, -1)]
            , dim=1)

            # inputs = torch.cat([local_ent_scores.view(n_ments * n_cands, -1),
            #                     torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1)], dim=1)

            scores = self.score_combine(inputs).view(n_ments, n_cands)

            return scores, self.actions

        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.actions[:]
        del self.order_saved_log_probs[:]
        del self.decision_order[:]
        del self.targets[:]

        if isTrain:
            self.flag += 1

        scores = None
        ment_sort_scores = None


        cumulative_mention_idxes = None
        cumulative_mention_mask = None

        cumulative_mention_mask_dyanmic = Variable(torch.FloatTensor(torch.ones(n_ments)).cuda())

        if isOrderLearning:
            if self.param_copy_switch and isSort == 'topic':
                self.ment2ment_mat_diag = copy.deepcopy(self.entity2entity_mat_diag)
                self.ment2ment_score_mat_diag = copy.deepcopy(self.entity2entity_score_mat_diag)

                self.ment2ment_mat_diag.requires_grad = False
                self.ment2ment_score_mat_diag.requires_grad = False

                # debug
                # print(self.entity2entity_mat_diag)
                # print(self.entity2entity_score_mat_diag)
                # print(self.ment2ment_mat_diag)
                # print(self.ment2ment_score_mat_diag)

                # deep copy parameters only one time
                self.param_copy_switch = False

            elif isSort == 'local':

                local_ment_scores = super(MulRelRanker, self).compute_local_similarity(token_ids, tok_mask,
                                                                                       ment_ids, ment_mask)
                # debug
                # print(local_ment_scores)
                # print(local_ment_scores.size())

                local_ment_avg_scores = local_ment_scores.view(n_ments, -1).mean(1)

                # print(local_ment_avg_scores)

                ment_avg_scores = local_ment_avg_scores.cpu().data.numpy()

                ment_sort_scores = ment_avg_scores.argsort()[::-1]

                # print(ment_sort_scores)
                # input()
        self.added_words = []
        self.added_ents = []

        cumulative_entity_ids = Variable(torch.LongTensor([self.ent_unk_id]).cuda())
        # print(cumulative_entity_ids)
        cumulative_knowledge_ids = Variable(torch.LongTensor([self.word_unk_id]).cuda())

        for i in range(n_ments):

            if isOrderLearning:
                if isSort == 'local':
                    if len(ment_sort_scores) != n_ments:
                        print("Sorting Error in Mention-Local Similarity!")
                        break

                    indx = int(ment_sort_scores[i])

                elif isSort == 'topic':
                    if i == 0:
                        # The start point always be the first one of the initial decision order
                        cumulative_mention_idxes = Variable(torch.LongTensor([0]).cuda())
                        # cumulative_mention_mask = Variable(torch.FloatTensor(torch.ones(n_ments)).cuda())
                        # cumulative_mention_mask[0] = 0.0

                        cumulative_mention_mask_dyanmic[0] = 0.0
                        cumulative_mention_mask = cumulative_mention_mask_dyanmic.clone()

                        indx = 0
                    else:
                        # --------------------- Old Order Learning Module --------------------------------
                        # var_indx = self.learning_order(cumulative_mention_idxes, cumulative_mention_mask,
                        #                                ment_ids, ment_mask, self.ment2ment_mat_diag,
                        #                                self.ment2ment_score_mat_diag, self.tok_top_n4ment)
                        #
                        # cumulative_mention_idxes = torch.cat([cumulative_mention_idxes, var_indx], dim=0)
                        # cumulative_mention_mask[var_indx] = 0.0
                        #
                        # indx = var_indx.data[0]
                        ##indx = var_indx
                        # --------------------------------------------------------------------------------

                        # --------------------- New Order Learning Module --------------------------------
                        ment_order_score = self.learning_order(cumulative_mention_idxes, cumulative_mention_mask,
                                                               ment_ids, ment_mask, self.ment2ment_mat_diag,
                                                               self.ment2ment_score_mat_diag, self.tok_top_n4ment)

                        # print('ment_order_score: ', ment_order_score)

                        order_action_prob = F.softmax(ment_order_score - torch.max(ment_order_score), 1)

                        # print('order_action_prob: ', order_action_prob)

                        if isOrderFixed:
                            val, order_action = torch.max(order_action_prob, 1)
                        else:
                            order_m = Categorical(order_action_prob)
                            order_action = order_m.sample()
                            self.order_saved_log_probs.append(order_m.log_prob(order_action))

                        # print('order_action: ', order_action)

                        cumulative_mention_idxes = torch.cat([cumulative_mention_idxes, order_action], dim=0)

                        # cumulative_mention_mask[order_action] = 0.0

                        cumulative_mention_mask_dyanmic[order_action] = 0.0
                        cumulative_mention_mask = cumulative_mention_mask_dyanmic.clone()

                        # print("cumulative_mention_idxes: ", cumulative_mention_idxes)
                        # print('cumulative_mention_mask: ', cumulative_mention_mask)

                        indx = order_action.data[0]
            else:
                indx = i

            self.decision_order.append(indx)
            self.targets.append(gold.data[indx][0])
            # print('indx: ', indx)

            # print('ent_coherence: ')
            ent_coherence, selected_ents = self.compute_coherence(cumulative_entity_ids, entity_ids[indx], entity_mask[indx],
                                                   self.entity2entity_mat_diag, self.entity2entity_score_mat_diag,
                                                   self.tok_top_n4ent, isWord=False)
            # print('kng_coherence: ')
            kng_coherence, selected_words = self.compute_coherence(cumulative_knowledge_ids, entity_ids[indx], entity_mask[indx],
                                                   self.knowledge2entity_mat_diag, self.knowledge2entity_score_mat_diag,
                                                   self.tok_top_n4word, isWord=True)
            # input()

            # Typing Feature
            inputs = torch.cat([sim[indx].view(n_cands, -1) if not basemodel == 'deeped' else local_ent_scores[indx].view(n_cands, -1),
                                #local_ent_scores[indx].view(n_cands, -1),
                                torch.log(p_e_m[indx] + 1e-20).view(n_cands, -1), ent_coherence.view(n_cands, -1),
                                tm[indx].view(n_cands, -1),
                                kng_coherence.view(n_cands, -1)], dim=1)

            # inputs = torch.cat([local_ent_scores[indx].view(n_cands, -1),
            #                     torch.log(p_e_m[indx] + 1e-20).view(n_cands, -1), ent_coherence.view(n_cands, -1),
            #                     kng_coherence.view(n_cands, -1)], dim=1)

            score = self.score_combine(inputs).view(1, n_cands)
            action_prob = F.softmax(score - torch.max(score), 1)

            if isTrain:
                if method == "SL":
                    # print("gold[indx]: ", gold[indx])
                    # print("gold.data[indx][0]: ", gold.data[indx][0])
                    # print("new               : ", entity_ids[indx][ [ gold.data[indx][0] ]])

                    new_entities = entity_ids[indx][ [ gold.data[indx][0]] ].unsqueeze(0)
                    cumulative_entity_ids = torch.cat([ cumulative_entity_ids, new_entities], dim=0)
                    cumulative_entity_ids = Variable(self.unique(cumulative_entity_ids.cpu().data.numpy()).cuda())

                    entity_idx = (entity_ids[indx][gold.data[0]]).item()
                    if entity_idx  in self.ent_inlinks:

                        external_inlinks = np.asarray(self.ent_inlinks[ entity_idx ][:self.tok_top_n4inlink])

                        cumulative_knowledge_ids = Variable(self.unique(
                            np.concatenate((cumulative_knowledge_ids.cpu().data.numpy(), external_inlinks), axis=0)).cuda())

                    # debug
                    # print("Ground Truth:", gold.data[indx][0])
                    #
                    # if (entity_ids[indx][gold.data[indx][0]]).data[0] in self.ent_inlinks:
                    #     print("Entity inLinks", self.ent_inlinks[(entity_ids[indx][gold.data[indx][0]]).data[0]])
                    #
                    # print("cumulative_entity_ids Size: ", cumulative_entity_ids)
                    # print("cumulative_knowledge_ids Size: ", cumulative_knowledge_ids)

                elif method == "RL":
                    m = Categorical(action_prob)
                    action = m.sample()

                    new_entities = entity_ids[indx][action.data[0]].unsqueeze(0)
                    cumulative_entity_ids = torch.cat([cumulative_entity_ids, new_entities], dim=0)
                    cumulative_entity_ids = Variable(self.unique(cumulative_entity_ids.cpu().data.numpy()).cuda())

                    entity_idx = (entity_ids[indx][action.data[0]]).item()
                    if entity_idx in self.ent_inlinks:
                        # external_inlinks = Variable(torch.LongTensor(self.ent_inlinks[(entity_ids[indx][action.data[0]]).data[0]][:self.tok_top_n4inlink]).cuda())
                        #
                        # cumulative_knowledge_ids = torch.cat([cumulative_knowledge_ids, external_inlinks], dim=0)
                        #
                        # cumulative_knowledge_ids = Variable(self.unique(cumulative_knowledge_ids.data).cuda)

                        external_inlinks = np.asarray(self.ent_inlinks[ entity_idx ][:self.tok_top_n4inlink])

                        cumulative_knowledge_ids = Variable(self.unique(
                            np.concatenate((cumulative_knowledge_ids.cpu().data.numpy(), external_inlinks), axis=0)).cuda())

                    # debug
                    # print("Ground Truth:", action.data[0])
                    #
                    # if (entity_ids[indx][action.data[0]]).data[0] in self.ent_inlinks:
                    #     print("Entity inLinks", self.ent_inlinks[(entity_ids[indx][action.data[0]]).data[0]])
                    #
                    # print("cumulative_entity_ids Size: ", cumulative_entity_ids)
                    # print("cumulative_knowledge_ids Size: ", cumulative_knowledge_ids)

                    # self.saved_log_probs.append(m.log_prob(action))
                    # print("cumulative_knowledge_ids Size: ", cumulative_knowledge_ids)

                    self.saved_log_probs.append(m.log_prob(action))
                    self.actions.append(action.data[0])

                    if action.data[0] == gold.data[indx][0]:
                        self.rewards.append(0)
                    else:
                        self.rewards.append(-1.)

            else:
                val, action = torch.max(action_prob, 1)

                # print("gold[action]: ", gold[action])
                # print("gold.data[action][0]: ", gold.data[action][0])
                if isDynamic == 0 or isDynamic == 1:
                    new_entities = entity_ids[indx][action.data[0]].unsqueeze(0)

                    cumulative_entity_ids = torch.cat([cumulative_entity_ids, new_entities], dim=0)
                    cumulative_entity_ids = Variable(self.unique(cumulative_entity_ids.cpu().data.numpy()).cuda())
                entities_idx = (entity_ids[indx][action.data[0]]).item()
                if isDynamic == 0 and entities_idx in self.ent_inlinks:

                    # external_inlinks = Variable(torch.LongTensor(self.ent_inlinks[(entity_ids[indx][action.data[0]]).data[0]][:self.tok_top_n4inlink]).cuda())
                    #
                    # cumulative_knowledge_ids = torch.cat([cumulative_knowledge_ids, external_inlinks], dim=0)
                    #
                    # cumulative_knowledge_ids = Variable(self.unique(cumulative_knowledge_ids.data))
                    external_inlinks = np.asarray(self.ent_inlinks[ entities_idx ][:self.tok_top_n4inlink])

                    cumulative_knowledge_ids = Variable(self.unique(np.concatenate((cumulative_knowledge_ids.cpu().data.numpy(), external_inlinks), axis=0)).cuda())

                if method == "RL":
                    self.actions.append(action.data[0])

                if isOrderLearning:
                    if action.data[0] == gold.data[indx][0]:
                        self.rewards.append(0)
                    else:
                        self.rewards.append(-1.)

                # if external_inlinks:
                #     self.added_words.append([idx for idx in external_inlinks.tolist()])
                # else:
                #     self.added_words.append([])
                #self.added_words.append([idx for idx in selected_words.data])
                #self.added_ents.append([idx for idx in selected_ents.data])
            if i == 0:
                scores = score
            else:
                scores = torch.cat([scores, score], dim=0)


        if isTrain and (len(self.rewards) != len(self.saved_log_probs) or len(self.rewards) != len(self.actions) or \
                len(self.actions) != len(self.saved_log_probs)):
            print("Running Error in RL Training!")
            print("Length for self.rewards: ", len(self.rewards))
            print("Length for self.saved_log_probs: ", len(self.saved_log_probs))
            print("Length for self.actions: ", len(self.actions))
            return

        if not isOrderFixed and isOrderLearning and len(self.decision_order) != len(self.order_saved_log_probs) + 1:
            print("Running Error in Order Learning!")
            print("Length for self.decision_order: ", len(self.decision_order))
            print("Length for self.order_saved_log_probs: ", len(self.order_saved_log_probs))
            return

        if isTrain and self.flag % 953 == 0:
            print(self.flag)
            print(self.decision_order)


        return scores, self.actions

    def learning_order(self, cumulative_idx, cumulative_mask, mention_ids, mention_mask, att_mat_diag,
                       score_att_mat_diag, window_size):
        # print('cumulative_idx', cumulative_idx)
        # print('cumulative_mask', cumulative_mask)

        n_cumulative_ments = cumulative_idx.size(0)
        n_ments = mention_ids.size(0)

        # mask all the unk_tokens and average token embeddings within the mention
        ment_vecs = self.word_embeddings(mention_ids)

        mention_mask = mention_mask.unsqueeze(-1)
        mention_mask = mention_mask.expand(mention_mask.size(0), mention_mask.size(1), self.emb_dims)

        # get the mention embeddings
        ment_vecs = (ment_vecs * mention_mask).mean(1)

        # get the disambiguated mentions' embeddings by indexing
        cumulative_ment_vecs = ment_vecs[cumulative_idx]

        cumulative_mat_mask = cumulative_mask.unsqueeze(1)

        # print(ment_vecs.size())
        # print(cumulative_ment_vecs.size())

        # att
        ment_tok_att_scores = torch.mm(ment_vecs * att_mat_diag, cumulative_ment_vecs.permute(1, 0))

        # print('ment_tok_att_scores', ment_tok_att_scores)

        ment_tok_att_scores = (ment_tok_att_scores * cumulative_mat_mask).add_((cumulative_mat_mask - 1).mul_(1e10))

        # print('ment_tok_att_scores', ment_tok_att_scores)

        tok_att_scores, _ = torch.max(ment_tok_att_scores, dim=0)

        # print('tok_att_scores', tok_att_scores)

        top_tok_att_scores, top_tok_att_ids = torch.topk(tok_att_scores, dim=0,
                                                         k=min(window_size, n_cumulative_ments))

        # print('top_tok_att_scores', top_tok_att_scores)
        # print('top_tok_att_ids', top_tok_att_ids)

        tok_att_probs = F.softmax(top_tok_att_scores, dim=0).view(-1, 1)
        tok_att_probs = tok_att_probs / torch.sum(tok_att_probs, dim=0, keepdim=True)

        # print('tok_att_probs', tok_att_probs)

        selected_tok_vecs = torch.gather(cumulative_ment_vecs, dim=0,
                                         index=top_tok_att_ids.view(-1, 1).repeat(1, cumulative_ment_vecs.size(1)))

        # print(selected_tok_vecs.size())
        # print((selected_tok_vecs * score_att_mat_diag).size())
        # print(tok_att_probs.size())
        # print(((selected_tok_vecs * score_att_mat_diag) * tok_att_probs).size())

        ctx_ment_vecs = torch.sum((selected_tok_vecs * score_att_mat_diag) * tok_att_probs, dim=0, keepdim=True)

        # print(ctx_ment_vecs.size())

        ment_ctx_scores = torch.mm(ment_vecs, ctx_ment_vecs.permute(1, 0)).view(-1, n_ments)

        # _, next_index = torch.max(ment_ctx_scores, dim=1)

        # print(ment_ctx_scores.size())

        scores = (ment_ctx_scores * cumulative_mask).add_((cumulative_mask - 1).mul_(1e10))

        # print('scores', scores)

        return scores

        # input()

        # _, next_index = torch.max(scores, dim=1)

        # print('next_index', next_index)

        # return next_index

    def unique(self, numpy_array):
        t = np.unique(numpy_array)
        return torch.from_numpy(t).type(torch.LongTensor)

    def compute_coherence(self, cumulative_ids, entity_ids, entity_mask, att_mat_diag, score_att_mat_diag, window_size, isWord=False):
        n_cumulative_entities = cumulative_ids.size(0)
        n_entities = entity_ids.size(0)

        if self.dca_method == 1 or self.dca_method == 2:
            window_size = 100

        # print('n_cumulative_entities', n_cumulative_entities)
        # print('n_entities', n_entities)

        try:
            if isWord:
                cumulative_entity_vecs = self.word_embeddings(cumulative_ids)
            else:
                cumulative_entity_vecs = self.entity_embeddings(cumulative_ids)
        except:
            print(cumulative_ids)
            input()

        # cumulative_entity_vecs = self.entity_embeddings(cumulative_ids)

        entity_vecs = self.entity_embeddings(entity_ids)

        # debug
        # print("Cumulative_entity_ids Size: ", cumulative_ids.size(), cumulative_ids.size(0))
        # print("Entity_ids Size: ", entity_ids.size(), entity_ids.size(0))
        # print("Cumulative_entity_vecs Size: ", cumulative_entity_vecs.size())
        # print("Entity_vecs Size: ", entity_vecs.size())

        # att
        ent2ent_att_scores = torch.mm(entity_vecs * att_mat_diag, cumulative_entity_vecs.permute(1, 0))
        ent_tok_att_scores, _ = torch.max(ent2ent_att_scores, dim=0)
        top_tok_att_scores, top_tok_att_ids = torch.topk(ent_tok_att_scores, dim=0, k=min(window_size, n_cumulative_entities))

        # print("Top_tok_att_scores Size: ", top_tok_att_scores.size())
        # print("Top_tok_att_scores: ", top_tok_att_scores)
        if self.dca_method == 2:
            entity_att_probs = F.softmax(top_tok_att_scores*0., dim=0).view(-1, 1)
        else:
            entity_att_probs = F.softmax(top_tok_att_scores, dim=0).view(-1, 1)
        entity_att_probs = entity_att_probs / torch.sum(entity_att_probs, dim=0, keepdim=True)

        # print("entity_att_probs: ", entity_att_probs)

        selected_tok_vecs = torch.gather(cumulative_entity_vecs, dim=0,
                                         index=top_tok_att_ids.view(-1, 1).repeat(1, cumulative_entity_vecs.size(1)))

        ctx_ent_vecs = torch.sum((selected_tok_vecs * score_att_mat_diag) * entity_att_probs, dim=0, keepdim=True)

        # print("Selected_vecs * diag Size: ", (selected_tok_vecs * att_mat_diag).size())
        # print("Before Sum Size: ", ((selected_tok_vecs * att_mat_diag) * entity_att_probs).size())
        # print("Ctx_ent_vecs Size: ", ctx_ent_vecs.size())

        ent_ctx_scores = torch.mm(entity_vecs, ctx_ent_vecs.permute(1, 0)).view(-1, n_entities)

        # print("Ent_ctx_scores", ent_ctx_scores)

        scores = (ent_ctx_scores * entity_mask).add_((entity_mask - 1).mul_(1e10))

        # print("Scores: ", scores)

        return scores, cumulative_ids[top_tok_att_ids.view(-1)].view(-1)

    def print_weight_norm(self):
        LocalCtxAttRanker.print_weight_norm(self)

        print('entity2entity_mat_diag', self.entity2entity_mat_diag.data.norm())
        print('entity2entity_score_mat_diag', self.entity2entity_score_mat_diag.data.norm())

        print('knowledge2entity_mat_diag', self.knowledge2entity_mat_diag.data.norm())
        print('knowledge2entity_score_mat_diag', self.knowledge2entity_score_mat_diag.data.norm())

        print('ment2ment_mat_diag', self.ment2ment_mat_diag.data.norm())
        print('ment2ment_score_mat_diag', self.ment2ment_score_mat_diag.data.norm())

        print('f - l1.w, b', self.score_combine[0].weight.data.norm(), self.score_combine[0].bias.data.norm())
        print('f - l2.w, b', self.score_combine[3].weight.data.norm(), self.score_combine[3].bias.data.norm())

        # print(self.ctx_layer[0].weight.data.norm(), self.ctx_layer[0].bias.data.norm())
        # print('relations', self.rel_embs.data.norm(p=2, dim=1))
        # X = F.normalize(self.rel_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).sqrt()
        # print(diff)
        #
        # print('ew_embs', self.ew_embs.data.norm(p=2, dim=1))
        # X = F.normalize(self.ew_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).sqrt()
        # print(diff)

    def regularize(self, max_norm=4):
        # super(MulRelRanker, self).regularize(max_norm)
        # print("----MulRelRanker Regularization----")

        l1_w_norm = self.score_combine[0].weight.norm()
        l1_b_norm = self.score_combine[0].bias.norm()
        l2_w_norm = self.score_combine[3].weight.norm()
        l2_b_norm = self.score_combine[3].bias.norm()

        if (l1_w_norm > max_norm).data.all():
            self.score_combine[0].weight.data = self.score_combine[0].weight.data * max_norm / l1_w_norm.data
        if (l1_b_norm > max_norm).data.all():
            self.score_combine[0].bias.data = self.score_combine[0].bias.data * max_norm / l1_b_norm.data
        if (l2_w_norm > max_norm).data.all():
            self.score_combine[3].weight.data = self.score_combine[3].weight.data * max_norm / l2_w_norm.data
        if (l2_b_norm > max_norm).data.all():
            self.score_combine[3].bias.data = self.score_combine[3].bias.data * max_norm / l2_b_norm.data

    def finish_episode(self, rewards_arr, log_prob_arr):
        if len(rewards_arr) != len(log_prob_arr):
            print("Size mismatch between Rwards and Log_probs!")
            return

        policy_loss = []
        rewards = []

        # we only give a non-zero reward when done
        g_return = sum(rewards_arr) / len(rewards_arr)

        # add the final return in the last step
        rewards.insert(0, g_return)

        R = g_return
        for idx in range(len(rewards_arr) - 1):
            R = R * self.gamma
            rewards.insert(0, R)

        rewards = torch.from_numpy(np.array(rewards)).type(torch.cuda.FloatTensor)

        for log_prob, reward in zip(log_prob_arr, rewards):
            policy_loss.append(-log_prob * reward)

        policy_loss = torch.cat(policy_loss).sum()

        return policy_loss

    def loss(self, scores, true_pos, method="SL", lamb=1e-7):
        loss = None

        # print("----MulRelRanker Loss----")
        if method == "SL":
            loss = F.multi_margin_loss(scores, true_pos, margin=self.margin)
        elif method == "RL":
            loss = self.finish_episode(self.rewards, self.saved_log_probs)

        # if self.use_local_only:
        #     return loss

        # regularization
        # X = F.normalize(self.rel_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).add_(1e-5).sqrt()
        # diff = diff * (diff < 1).float()
        # loss -= torch.sum(diff).mul(lamb)
        #
        # X = F.normalize(self.ew_embs)
        # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).add_(1e-5).sqrt()
        # diff = diff * (diff < 1).float()
        # loss -= torch.sum(diff).mul(lamb)
        return loss

    def order_loss(self):
        return self.finish_episode(self.rewards[1:], self.order_saved_log_probs)

    def get_order_truth(self):
        return self.decision_order, self.targets

    def _calc(self, h, t, r):
        score = (h * r) * t
        return torch.sum(score, -1)

    def score(self, h, t, r):
        score = (h * r) * t

        return -torch.sum(score, -1)

    def encode(self, indicies):
        return self.entity_embeddings(indicies)        

    def extract_rel(self, r):
        return self.rel_embeddings(r)

    def calculate_loss(self, pos_pair, neg_pair):
        h, r, t = pos_pair

        neg_h, neg_r, neg_t = neg_pair

        p_score = self._calc(*self.kg_forward(h, r, t))
        n_score = self._calc(*self.kg_forward(neg_h, neg_r, neg_t))

        return self.criterion(p_score, n_score).mean()

    def self_regularization(self):
        return (self.entity_embeddings.weight.norm(p = 3)**3 + \
            self.rel_embeddings.weight.norm(p = 3)**3 + \
            self.type_embeddings.weight.norm(p = 3)**3 )

    def calculate_loss_avg(self, type_triples):
        h, r, t_types, neg_t_types = type_triples

        t_types_emb = self.type_embeddings(t_types)

        neg_t_types_emb = self.type_embeddings(neg_t_types)


        p_score = self._calc(self.entity_embeddings(h), t_types_emb.mean(1), self.rel_embeddings(r))
        n_score = self._calc(self.entity_embeddings(h), neg_t_types_emb.mean(1), self.rel_embeddings(r))

        return self.criterion(p_score, n_score),  (
            diversity_regularization( neg_t_types_emb.view(-1, self.hidden_size)) + \
            diversity_regularization( t_types_emb.view(-1, self.hidden_size) )
            ) / 2, self.type_regularization( (h, r, t_types) )

    def kg_regularization(self, data):
        batch_h, batch_r, batch_t  = data
        h = self.entity_embeddings(batch_h)
        t = self.entity_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def type_regularization(self, data):
        batch_h, batch_r, batch_t  = data
        h = self.entity_embeddings(batch_h)
        t = self.type_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def kg_forward(self, h, r, t):
        h = self.entity_embeddings(h)
        t = self.entity_embeddings(t)
        r = self.rel_embeddings(r)
        return h, t, r
