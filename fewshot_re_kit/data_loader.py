import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, pid2name, encoder, N, K, Q, root, ispubmed=False, rel2id=None, id2rel=None, seed=None):
        self.root = root
        random.seed(seed)
        path = os.path.join(root, name + ".json")
        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file does not exist!")
            assert 0
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        # self.rel2id,self.id2rel=self.get_id(self.pid2name)
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder
        self.rel2id = rel2id
        self.id2rel = id2rel

    def get_id(self, pid2name):
        rel2id = {}
        for rel in pid2name.keys():
            rel2id[rel] = len(rel2id)
        id2rel = {i: j for j, i in rel2id.items()}

        return rel2id, id2rel

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask

    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __additem__(self, d, word, pos1, pos2, mask, class_name):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['rel'].append(torch.tensor(self.rel2id[class_name]).long())

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel': []}
        query_label = []
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask, class_name)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask, class_name)
                count += 1

            query_label += [i] * self.Q

        return support_set, query_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel': []}
    batch_label = []
    support_sets, query_sets, query_labels= zip(*data)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]

    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label


def get_loader(name, pid2name, encoder, N, K, Q, batch_size,
               num_workers=8, collate_fn=collate_fn, ispubmed=False, root='./data', rel2id=None, id2rel=None,
               seed=None):
    dataset = FewRelDataset(name, pid2name, encoder, N, K, Q, root, ispubmed, rel2id, id2rel, seed=seed)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


def get_rel2id(root, paths):
    rel2id = {}
    for path_ in paths:
        path = os.path.join(root, path_ + ".json")
        data_ = json.load(open(path))
        for rel in data_.keys():
            if rel not in rel2id.keys():
                rel2id[rel] = len(rel2id)
    id2rel = {i: j for j, i in rel2id.items()}

    return rel2id, id2rel
