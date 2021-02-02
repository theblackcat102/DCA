import re
import random
from collections import OrderedDict, defaultdict
from pprint import pprint
import pickle as pkl
import json
from torch.utils.data import Dataset
import os
import torch
from vocabulary import Vocabulary
import numpy as np

doc2type = pkl.load(open('./data/doc2type.pkl', 'rb'))
entity2type = pkl.load(open('./data/entity2type.pkl', 'rb'))
mtype2id = {'PER':0, 'ORG':1, 'GPE':2, 'UNK':3}

def judge(s1, s2):
    if s1==s2:
        return True
    if s2.replace('. ', ' ').replace('.', ' ') == s1:
        return True
    if s2.replace('-', ' ') == s1:
        return True
    return False
def read_csv_file(path):
    data = {}
    flag = 0

    if path.find('aida')>=0:
        flag = 1
    else:
        types = json.load(open('./data/generated/type/'+path.split('/')[-1].split('.')[0]+'.json', 'rb'))

    docid = '0'
    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            comps = line.strip().split('\t')
            doc_name = comps[0] + ' ' + comps[1]
            mention = comps[2]
            mtype = [0,0,0,0]
            if flag == 1:
                doc = ''
                for c in doc_name:
                    try:
                        doc += str(int(c))
                    except:
                        break
                if not doc==docid:
                    docid = doc
                    p = 0
                    tt = doc2type[docid]
                try:

                    while not judge(mention.lower(), tt[p][0].lower()):
                        p += 1
                    mtype[mtype2id[tt[p][1]]] = 1

                except:
                    print(docid+mention)

                    mtype[mtype2id['UNK']] = 1
            else:
                if path.find('wikipedia')<0:
                    tt = types['sample_%d'%i]['pred'] + types['sample_%d'%i]['overlap']
                    for t in tt:
                        if t == 'MISC':
                            t = 'UNK'
                        if t == 'LOC':
                            t = 'GPE'
                        mtype[mtype2id[t]] = 1
                else:
                    mtype[mtype2id['UNK']] = 1

            lctx = comps[3]
            rctx = comps[4]

            if comps[6] != 'EMPTYCAND':
                cands = [c.split(',') for c in comps[6:-2]]
                cands = [[','.join(c[2:]).replace('"', '%22').replace(' ', '_'), float(c[1])] for c in cands]
            else:
                cands = []

            gold = comps[-1].split(',')
            if gold[0] == '-1':
                gold = (','.join(gold[2:]).replace('"', '%22').replace(' ', '_'), 1e-5, -1)
            else:
                gold = (','.join(gold[3:]).replace('"', '%22').replace(' ', '_'), 1e-5, -1)

            if doc_name not in data:
                data[doc_name] = []
            data[doc_name].append({'mention': mention,
                                   'mtype': mtype,
                                   'context': (lctx, rctx),
                                   'candidates': cands,
                                   'gold': gold})
    return data


def load_person_names(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(line.strip().replace(' ', '_'))
    return set(data)


def with_coref(dataset, person_names):
    for doc_name, content in dataset.items():
        for cur_m in content:
            coref = find_coref(cur_m, content, person_names)
            if coref is not None and len(coref) > 0:
                cur_cands = {}
                for m in coref:
                    for c, p in m['candidates']:
                        cur_cands[c] = cur_cands.get(c, 0) + p
                for c in cur_cands.keys():
                    cur_cands[c] /= len(coref)
                cur_m['candidates'] = sorted(list(cur_cands.items()), key=lambda x: x[1])[::-1]

    for data_name, content in dataset.items():
        for cur_m in content:
            for i, cand in enumerate(cur_m['candidates']):
                cur_m['candidates'][i] = list(cand)
                cur_m['candidates'][i].append([0, 0, 0, 0])
                if cur_m['candidates'][i][0] in entity2type and len(entity2type[cur_m['candidates'][i][0]]) > 0:
                    for t in entity2type[cand[0]]:
                        cur_m['candidates'][i][-1][mtype2id[t]] = 1
                else:
                    cur_m['candidates'][i][-1][-1] = 1
            for cand in cur_m['candidates']:
                assert len(cand) == 3


def find_coref(ment, mentlist, person_names):
    cur_m = ment['mention'].lower()
    coref = []
    for m in mentlist:
        if len(m['candidates']) == 0 or m['candidates'][0][0] not in person_names:
            continue

        mention = m['mention'].lower()
        start_pos = mention.find(cur_m)
        if start_pos == -1 or mention == cur_m:
            continue

        end_pos = start_pos + len(cur_m) - 1
        if (start_pos == 0 or mention[start_pos-1] == ' ') and \
                (end_pos == len(mention) - 1 or mention[end_pos + 1] == ' '):
            coref.append(m)

    return coref


def read_conll_file(data, path):
    conll = {}
    with open(path, 'r', encoding='utf8') as f:
        cur_sent = None
        cur_doc = None

        for line in f:
            line = line.strip()
            if line.startswith('-DOCSTART-'):
                docname = line.split()[1][1:]
                conll[docname] = {'sentences': [], 'mentions': []}
                cur_doc = conll[docname]
                cur_sent = []

            else:
                if line == '':
                    cur_doc['sentences'].append(cur_sent)
                    cur_sent = []

                else:
                    comps = line.split('\t')
                    tok = comps[0]
                    cur_sent.append(tok)

                    if len(comps) >= 6:
                        bi = comps[1]
                        wikilink = comps[4]
                        if bi == 'I':
                            cur_doc['mentions'][-1]['end'] += 1
                        else:
                            new_ment = {'sent_id': len(cur_doc['sentences']),
                                        'start': len(cur_sent) - 1,
                                        'end': len(cur_sent),
                                        'wikilink': wikilink}
                            cur_doc['mentions'].append(new_ment)

    # merge with data
    rmpunc = re.compile('[\W]+')
    for doc_name, content in data.items():
        conll_doc = conll[doc_name.split()[0]]
        content[0]['conll_doc'] = conll_doc

        cur_conll_m_id = 0
        for m in content:
            mention = m['mention']
            # flag = 0

            while True:
                cur_conll_m = conll_doc['mentions'][cur_conll_m_id]
                cur_conll_mention = ' '.join(conll_doc['sentences'][cur_conll_m['sent_id']][cur_conll_m['start']:cur_conll_m['end']])
                if rmpunc.sub('', cur_conll_mention.lower()) == rmpunc.sub('', mention.lower()):
                    m['conll_m'] = cur_conll_m

                    # if flag == 1:
                    #     print(cur_conll_m_id, cur_conll_mention, mention)
                    # flag = 0

                    cur_conll_m_id += 1
                    break
                else:
                    # print(cur_conll_m_id, cur_conll_mention, mention)
                    # flag = 1
                    cur_conll_m_id += 1


def reorder_dataset(data, order):
    # the default order is "offset"
    if order == "random" or order == "size":
        for doc_name, content in data.items():
            conll_doc = content[0]['conll_doc']

            if order == "random":
                random.shuffle(data[doc_name])
            elif order == "size":
                data[doc_name] = sorted(data[doc_name], key=lambda x: len(x['candidates']))

            data[doc_name][0]['conll_doc'] = conll_doc


def curriculum_reorder(data):
    sorted_by_value = sorted(data.items(), key=lambda kv: len(kv[1]))

    data_ordered = OrderedDict()
    for doc_name, content in sorted_by_value:
        data_ordered[doc_name] = content

    return data_ordered


def eval(testset, system_pred):
    gold = []
    pred = []

    for doc_name, content in testset.items():
        gold += [c['gold'][0] for c in content]
        pred += [c['pred'][0] for c in system_pred[doc_name]]

    true_pos = 0
    for g, p in zip(gold, pred):
        if g == p and p != 'NIL':
            true_pos += 1

    precision = true_pos / len([p for p in pred if p != 'NIL'])
    recall = true_pos / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


class CoNLLDataset:
    """
    reading dataset from CoNLL dataset, extracted by https://github.com/dalab/deep-ed/
    """

    def __init__(self, path, conll_path, person_path, order, method):
        print('load csv')
        self.train = read_csv_file(path + '/aida_train.csv')
        self.testA = read_csv_file(path + '/aida_testA.csv')
        self.testB = read_csv_file(path + '/aida_testB.csv')
        self.msnbc = read_csv_file(path + '/wned-msnbc.csv')
        self.ace2004 = read_csv_file(path + '/wned-ace2004.csv')
        self.aquaint = read_csv_file(path + '/wned-aquaint.csv')
        self.clueweb = read_csv_file(path + '/wned-clueweb.csv')
        self.wikipedia = read_csv_file(path + '/wned-wikipedia.csv')
        self.wikipedia.pop('Jiří_Třanovský Jiří_Třanovský', None)

        print('process coref')
        person_names = load_person_names(person_path)
        with_coref(self.train, person_names)
        with_coref(self.testA, person_names)
        with_coref(self.testB, person_names)
        with_coref(self.msnbc, person_names)
        with_coref(self.ace2004, person_names)
        with_coref(self.aquaint, person_names)
        with_coref(self.clueweb, person_names)
        with_coref(self.wikipedia, person_names)

        print('load conll')
        read_conll_file(self.train, conll_path + '/AIDA/aida_train.txt')
        read_conll_file(self.testA, conll_path + '/AIDA/testa_testb_aggregate_original')
        read_conll_file(self.testB, conll_path + '/AIDA/testa_testb_aggregate_original')
        read_conll_file(self.msnbc, conll_path + '/wned-datasets/msnbc/msnbc.conll')
        read_conll_file(self.ace2004, conll_path + '/wned-datasets/ace2004/ace2004.conll')
        read_conll_file(self.aquaint, conll_path + '/wned-datasets/aquaint/aquaint.conll')
        read_conll_file(self.clueweb, conll_path + '/wned-datasets/clueweb/clueweb.conll')
        read_conll_file(self.wikipedia, conll_path + '/wned-datasets/wikipedia/wikipedia.conll')

        print('reorder mentions within the dataset')
        reorder_dataset(self.train, order)
        reorder_dataset(self.testA, order)
        reorder_dataset(self.testB, order)
        reorder_dataset(self.msnbc, order)
        reorder_dataset(self.ace2004, order)
        reorder_dataset(self.aquaint, order)
        reorder_dataset(self.clueweb, order)
        reorder_dataset(self.wikipedia, order)


def negsamp_vectorized_bsearch(pos_inds, n_items, n_samp=32, items=None):
    """ Guess and check vectorized
    Assumes that we are allowed to potentially 
    return less than n_samp samples
    """
    if items is not None:
        raw_samps_idx = np.random.randint(0, len(items)-1, size=n_samp)
        raw_samps = items[raw_samps_idx]
    else:
        raw_samps = np.random.randint(0, n_items, size=n_samp)


    if len(pos_inds) > 0:
        ss = np.searchsorted(pos_inds, raw_samps)
        pos_mask = raw_samps == np.take(pos_inds, ss, mode='clip')
        neg_inds = raw_samps[~pos_mask]
        return neg_inds
    return raw_samps


class KGDataset(Dataset):
    type_names = [
            'http://purl.org/dc/terms/subject',
            'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    ]

    def __init__(self, data_path='data', max_type_size=5):
        self.max_type_size = max_type_size
        entity2id_path = os.path.join(data_path,'generated/embeddings/word_ent_embs/dict.entity')
        self.entity_dict = Vocabulary.load(entity2id_path).word2id
        triplets = pkl.load(open(os.path.join(data_path, 'triplets.pkl'), 'rb'))
        type_triplets = pkl.load(open(os.path.join(data_path, 'type_triplets.pkl'), 'rb'))
        self.type_dict = pkl.load(open(os.path.join(data_path, 'type2id.pkl'), 'rb'))
        self.rel_dict = pkl.load(open(os.path.join(data_path, 'rel2id.pkl'), 'rb'))
        rel2id = {}
        # if os.path.join(data_path, 'clean_triplets.pkl'):
        self.triplets = []

        for (h, r, t) in triplets:
            if h in self.entity_dict and t in self.entity_dict:
                self.triplets.append((h, r, t))
        # else:
        #     pkl.load(os.path.join(data_path, 'clean_triplets.pkl'))
        type_triplets_map = defaultdict(list)
        for (h, r, type_) in type_triplets:
            type_triplets_map[h+'\t'+r].append(type_)
        
        self.type_triplets = []
        for key, types in type_triplets_map.items():
            if len(types) >= max_type_size:
                h, r = key.split('\t')
                if h in self.entity_dict:
                    self.type_triplets.append((self.entity_dict[h], self.rel_dict[r], np.array([ self.type_dict[t] for t in types ]) ))

        self.ent_size = len(self.entity_dict)
        self.rel_size = len(self.rel_dict)
        self.type_size = len(self.type_dict)

        self.num_entries = len(self.triplets)
 
    def __len__(self):
        return self.num_entries


    def __getitem__(self, index):
        (h, r, t) = self.triplets[index]
        h_idx = self.entity_dict[h]
        r_idx = self.rel_dict[r]
        t_idx = self.entity_dict[t]
        pos_triplet = (h_idx, r_idx, t_idx)
        neg_triplet = self.sample_negative(*pos_triplet)

        type_idx = random.randint(0, len(self.type_triplets)-1)
        type_triplets = self.type_triplets[type_idx]
        valid_types = type_triplets[2]

        negative_type_triplets = self.sample_negative_types(valid_types)[:self.max_type_size ]
        random_idx = torch.randint(len(valid_types), (self.max_type_size, )).flatten()
        # print(random_idx, valid_types)
        output_types = ( type_triplets[0], type_triplets[1], 
            torch.from_numpy(valid_types[random_idx]).long(), negative_type_triplets.long() )


        return pos_triplet, neg_triplet, output_types

    def sample_negative_types(self, t):
        pos_ids = t

        negative_samples = torch.from_numpy(negsamp_vectorized_bsearch(pos_ids, 
            self.type_size, n_samp=len(pos_ids)*5).flatten())

        return negative_samples


    def sample_negative(self, h, r, t):
        replace_head = random.random() > 0.5
        tmp = random.randint(0, self.ent_size-1)
        while tmp == h or tmp == r:
            tmp = random.randint(0, self.ent_size-1)
        if replace_head:
            return tmp, r, t
        return h, r, tmp

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = KGDataset()
    dataloader = DataLoader(dataset, batch_size=1024, num_workers=5)
    # dataset[0]
    for batch in dataloader:
        print(batch[0][0].shape)