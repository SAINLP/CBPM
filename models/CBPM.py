import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import nn
from torch.nn import functional as F
import pickle
from sklearn.cluster import KMeans


class Auto_Alpha(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fc = nn.Linear(1536 * 2, self.opt.mlp_hidden_state)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.opt.mlp_hidden_state, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, support, center):
        ipt = torch.cat([support, center], dim=-1)
        output = self.fc(ipt)
        output = self.dropout(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output


class CBPM(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len, hierarchical=None, rel2id=None, id2rel=None, opt=None):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.opt = opt
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.hierarchical = hierarchical
        if self.hierarchical is not None:
            self.rel_sim = pickle.load(open(self.hierarchical, "rb"))
        self.rel2id = rel2id
        self.id2rel = id2rel
        self.auto_alpha = Auto_Alpha(opt)

    def __batch_dist__(self, S, Q):
        return -torch.sqrt((torch.pow(S.unsqueeze(1) - Q.unsqueeze(2), 2)).sum(3))

    def get_h(self, x, y):
        h = [0 for i in range(len(y))]
        for i, yy in enumerate(y):
            h[i] = 1 - self.rel_sim[self.id2rel[x]][self.id2rel[yy]]
        return h

    def unique_(self, tensor):
        uni = []
        for i in tensor.tolist():
            if i not in uni:
                uni.append(i)
        return torch.tensor(uni)

    def get_rel_dist(self, B, N, rel):
        rel = self.unique_(rel)
        frame = self.get_hierarchical_frame(B, N)
        frame = rel[frame].view(-1, N)
        return_h = []
        for x, y in zip(rel.tolist(), frame.tolist()):
            return_h.append(self.get_h(x, y))
        return torch.tensor(return_h)

    def get_hierarchical_frame(self, B, N):
        frame = torch.zeros(B, N, N)
        for b in range(B):
            for i in range(N):
                index_ori = [i for i in range(b * N, (b + 1) * N)]
                positive = index_ori[i]
                index_ori.pop(i)
                index_ori = [positive] + index_ori
                frame[b, i] = torch.tensor(index_ori)
        return frame.long()

    def __modify_support__(self, support_mean, query, kmeans):
        p_labels = kmeans.labels_
        centers = torch.stack([query[p_labels == i].mean(dim=0) for i in range(support_mean.shape[0])])
        support2center = torch.argmax(-(torch.pow(centers.unsqueeze(0) - support_mean.unsqueeze(1), 2)).sum(-1), dim=1)
        alpha = self.auto_alpha(support_mean, centers)
        modify_support = (1. - alpha) * support_mean + alpha * centers[support2center]
        return modify_support

    def query_km(self, support, query, N):
        support_mean = torch.mean(support, 2)
        sp = [self.__modify_support__(support_mean[i], query[i],
                                      KMeans(n_clusters=N, random_state=10).fit(query[i].detach().cpu())) for i in
              range(query.shape[0])]
        return torch.stack(sp)

    def forward(self, support, query, N, K, total_Q, is_eval=False, record=False):
        support_emb = self.sentence_encoder(support)
        query_emb = self.sentence_encoder(query)
        query_emb_temp = self.sentence_encoder(query)
        embedding_dim = support_emb.shape[1]
        Q = int(total_Q / N)
        support_emb = support_emb.view(-1, N, K, embedding_dim)
        query_emb = query_emb.view(-1, total_Q, embedding_dim)
        query_emb_temp = query_emb_temp.view(-1, total_Q, embedding_dim)
        B = support_emb.shape[0]
        proto_glo = self.query_km(support_emb, query_emb_temp, N)
        logits = self.__batch_dist__(proto_glo, query_emb)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        sim = None
        if not is_eval:
            m = self.get_rel_dist(B, N, query['rel'])
            sim = torch.repeat_interleave(m, Q, dim=0)
        return logits, pred, sim