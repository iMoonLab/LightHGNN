import dhg
import torch
import numpy as np
from collections import defaultdict
from dhg.data import Cora, Pubmed, Citeseer
from dhg.data import CoauthorshipCora, CoauthorshipDBLP
from dhg.data import CocitationCora, CocitationPubmed, CocitationCiteseer
from dhg.data import News20, DBLP4k, IMDB4k, Recipe100k, Recipe200k


class MultiExpMetric:
    def __init__(self):
        self.t = defaultdict(list)
        self.s = defaultdict(list)

    def update(self, res):
        self._update(self.t, res['t'])
        self._update(self.s, res['s'])

    def _update(self, data, new_res):
        for k, v in new_res.items():
            data[k].append(v)

    def __str__(self, ):
        ret = []
        ret.append('Teacher:')
        for k, v in self.t.items():
            v = np.array(v)
            ret.append(f"\t{k} -> {v.mean():.5f} - {v.std():.5f}")
        ret.append('Student:')
        for k, v in self.s.items():
            v = np.array(v)
            ret.append(f"\t{k} -> {v.mean():.5f} - {v.std():.5f}")
        return '\n'.join(ret)

def load_data(name):
    if name == 'cora':
        data = Cora()
        edge_list = data['edge_list']
    elif name == 'pubmed':
        data = Pubmed()
        edge_list = data['edge_list'] 
    elif name == 'citeseer':
        data = Citeseer()
        edge_list = data['edge_list'] 
    if name == 'ca_cora':
        data = CoauthorshipCora()
        edge_list = data['edge_list']
    elif name == 'coauthorship_dblp':
        data = CoauthorshipDBLP()
        edge_list = data['edge_list']
    elif name == 'cc_cora':
        data = CocitationCora()
        edge_list = data['edge_list']
    elif name == 'cc_citeseer':
        data = CocitationCiteseer()
        edge_list = data['edge_list']
    elif name == 'news20':
        data = News20()
        edge_list = data['edge_list']
    elif name == 'dblp4k_paper':
        data = DBLP4k()
        edge_list = data['edge_by_paper']
    elif name == 'dblp4k_term':
        data = DBLP4k()
        edge_list = data['edge_by_term']
    elif name == 'dblp4k_conf':
        data = DBLP4k()
        edge_list = data['edge_by_conf']
    elif name == 'imdb_aw':
        data = IMDB4k()
        edge_list = data['edge_by_actor'] + data['edge_by_director']
    elif name == 'recipe_100k':
        data = Recipe100k()
        edge_list = data['edge_list']
    elif name == 'recipe_200k':
        data = Recipe200k()
        edge_list = data['edge_list']
    else:
        raise NotImplementedError
    return data, edge_list


def product_split(train_mask, val_mask, test_mask, test_ind_ratio):
    train_idx, val_idx, test_idx = torch.where(train_mask)[0], torch.where(val_mask)[0], torch.where(test_mask)[0]
    test_idx_shuffle = torch.randperm(len(test_idx))
    num_ind = int(len(test_idx) * test_ind_ratio)
    test_ind_idx, test_tran_idx = test_idx[test_idx_shuffle[:num_ind]], test_idx[test_idx_shuffle[num_ind:]]
    obs_idx = torch.cat([train_idx, val_idx, test_tran_idx]).numpy().tolist()

    num_obs, num_train, num_val = len(obs_idx), len(train_idx), len(val_idx)
    test_ind_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    obs_train_mask = torch.zeros(num_obs, dtype=torch.bool)
    obs_val_mask = torch.zeros(num_obs, dtype=torch.bool)
    obs_test_mask = torch.zeros(num_obs, dtype=torch.bool)

    test_ind_mask[test_ind_idx] = True
    obs_train_mask[:num_train] = True
    obs_val_mask[num_train:num_train+num_val] = True
    obs_test_mask[num_train+num_val:] = True
    return obs_idx, obs_train_mask, obs_val_mask, obs_test_mask, test_ind_mask 


def re_index(vec):
    res = vec.clone()
    raw_id, new_id = res[0].item(), 0
    for idx in range(len(vec)):
        if vec[idx].item() != raw_id:
            raw_id, new_id = vec[idx].item(), new_id + 1
        res[idx] = new_id
    return res


def sub_hypergraph(hg: dhg.Hypergraph, v_idx):
    v_map = {v: idx for idx, v in enumerate(v_idx)}
    v_set = set(v_idx)
    e_list, w_list = [], []
    for e, w in zip(*hg.e):
        new_e = []
        for v in e:
            if v in v_set:
                new_e.append(v_map[v])
        if len(new_e) >= 1:
            e_list.append(tuple(new_e))
            w_list.append(w)
    return dhg.Hypergraph(len(v_set), e_list, w_list)


def fix_iso_v(G: dhg.Hypergraph):
    # fix isolated vertices
    iso_v = np.array(G.deg_v)==0
    if np.any(iso_v):
        extra_e = [tuple([e, ]) for e in np.where(iso_v)[0]]
        G.add_hyperedges(extra_e)
    return G


def ho_topology_score(X, G: dhg.Hypergraph):
    if isinstance(G, dhg.Graph):
        G = dhg.Hypergraph.from_graph(G)
    e_s = []
    X_e = G.v2e(X, aggr='mean')
    for e_idx in range(G.num_e):
        cur_s = []
        for v_idx in G.nbr_v(e_idx):
            cur_s.append(torch.norm(X_e[e_idx] - X[v_idx], p=2).item())
        e_s.append(np.mean(cur_s))
    return np.mean(e_s)
