import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import hydra
import logging
import numpy as np
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from utils import load_data, fix_iso_v, ho_topology_score
from my_model import MyGCN, MyHGNN, MyMLPs

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dhg.nn import MLP
from dhg import Hypergraph, Graph
from dhg.random import set_seed
from dhg.utils import split_by_num
from dhg.models import HGNNP, HGNN, HNHN, UniGCN, UniGAT, GCN, GAT
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


# =========================================================================
# train teacher
def train(net, X, G, lbls, train_mask, optimizer):
    net.train()
    optimizer.zero_grad()
    outs = net(X, G)
    loss = F.nll_loss(F.log_softmax(outs[train_mask], dim=1), lbls[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def valid(net, X, G, lbls, mask, evaluator):
    net.eval()
    outs = net(X, G)
    res = evaluator.validate(lbls[mask], outs[mask])
    return res


@torch.no_grad()
def test(net, X, G, lbls, mask, evaluator, ft_noise_level=0):
    net.eval()
    if ft_noise_level > 0:
        X = (1 - ft_noise_level) * X + ft_noise_level * torch.randn_like(X)
    outs = net(X, G)
    res = evaluator.test(lbls[mask], outs[mask])
    return res


# =========================================================================
# train student
class HighOrderConstraint(nn.Module):
    def __init__(self, model, X, G, noise_level=1.0, tau=1.0):
        super().__init__()
        model.eval()
        self.tau = tau
        pred = model(X, G).softmax(dim=-1).detach()
        entropy_x = -(pred * pred.log()).sum(1, keepdim=True)
        entropy_x[entropy_x.isnan()] = 0
        entropy_e = G.v2e(entropy_x, aggr="mean")

        X_noise = X.clone() * (torch.randn_like(X) + 1) * noise_level
        pred_ = model(X_noise, G).softmax(dim=-1).detach()
        entropy_x_ = -(pred_ * pred_.log()).sum(1, keepdim=True)
        entropy_x_[entropy_x_.isnan()] = 0
        entropy_e_ = G.v2e(entropy_x_, aggr="mean")

        self.delta_e_ = (entropy_e_ - entropy_e).abs()
        self.delta_e_ = 1 - self.delta_e_ / self.delta_e_.max()
        self.delta_e_ = self.delta_e_.squeeze()

    def forward(self, pred_s, pred_t, G):
        pred_s, pred_t = F.softmax(pred_s, dim=1), F.softmax(pred_t, dim=1)
        e_mask = torch.bernoulli(self.delta_e_).bool()
        pred_s_e = G.v2e(pred_s, aggr="mean")
        pred_s_e = pred_s_e[e_mask]
        pred_t_e = G.v2e(pred_t, aggr="mean")
        pred_t_e = pred_t_e[e_mask]
        loss = F.kl_div(torch.log(pred_s_e / self.tau), pred_t_e / self.tau, reduction="batchmean", log_target=True)
        return loss


def train_stu(net, X, G, lbls, out_t, train_mask, optimizer, hc=None, lamb=0):
    net.train()
    optimizer.zero_grad()
    outs = net(X)
    loss_x = F.nll_loss(F.log_softmax(outs[train_mask], dim=1), lbls[train_mask])
    loss_k = F.kl_div(F.log_softmax(outs, dim=1), F.softmax(out_t, dim=1), reduction="batchmean", log_target=True)
    if hc is not None:
        loss_h = hc(outs, out_t, G)
        loss_k = loss_h + loss_k
    loss = loss_x * lamb + loss_k * (1 - lamb)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def valid_stu(net, X, lbls, mask, evaluator):
    net.eval()
    outs = net(X)
    res = evaluator.validate(lbls[mask], outs[mask])
    return res


@torch.no_grad()
def test_stu(net, X, lbls, mask, evaluator, ft_noise_level=0):
    net.eval()
    if ft_noise_level > 0:
        X = (1 - ft_noise_level) * X + ft_noise_level * torch.randn_like(X)
    outs = net(X)
    res = evaluator.test(lbls[mask], outs[mask])
    return res


# =========================================================================
def exp(seed, cfg: DictConfig):
    set_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data, edge_list = load_data(cfg.data.name)

    if cfg.model.teacher in ['gcn', 'gat']: # 图模型
        if cfg.data.name in ['cora', 'pubmed', 'citeseer']: # 图数据集
            G = Graph(data["num_vertices"], edge_list)
        else: # 超图数据集
            g = Hypergraph(data["num_vertices"], edge_list)
            G = Graph.from_hypergraph_clique(g)
        G.add_extra_selfloop()
    else: # 超图模型
        if cfg.data.name in ['cora', 'pubmed', 'citeseer']: # 图数据集
            g = Graph(data["num_vertices"], edge_list)
            G = Hypergraph.from_graph(g)
            G.add_hyperedges_from_graph_kHop(g, 1)
        else: # 超图数据集
            G = Hypergraph(data["num_vertices"], edge_list)
        G = fix_iso_v(G)
    train_mask, val_mask, test_mask = split_by_num(
        data["num_vertices"], data["labels"], cfg.data.num_train, cfg.data.num_val
    )
    X, lbl = data["features"], data["labels"]

    if cfg.model.teacher == "hgnn":
        # net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=False)
        net = MyHGNN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.teacher == "hgnnp":
        net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.teacher == "hnhn":
        net = HNHN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.teacher == "unigcn":
        net = UniGCN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.teacher == "unigat":
        net = UniGAT(X.shape[1], 8, data["num_classes"], 4, use_bn=False)
    elif cfg.model.teacher == "gcn":
        # net = GCN(X.shape[1], 32, data["num_classes"], use_bn=False)
        net = MyGCN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.teacher == "gat":
        net = GAT(X.shape[1], 8, data["num_classes"], num_heads=4, use_bn=False)
    else:
        raise NotImplementedError

    # train teacher
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    X, lbl, G = X.to(device), lbl.to(device), G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, G, lbl, train_mask, optimizer)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = valid(net, X, G, lbl, val_mask, evaluator)
            if val_res > best_val:
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    # test
    net.load_state_dict(best_state)
    res_t = test(net, X, G, lbl, test_mask, evaluator, cfg.data.ft_noise_level)
    logging.info(f"teacher test best epoch: {best_epoch}, res: {res_t}")

    # -------------------------------------------------------------------------------------
    # train student
    out_t = net(X, G).detach()
    if cfg.model.student == "light_hgnnp":
        hc = HighOrderConstraint(net, X, G, noise_level=cfg.data.hc_noise_level, tau=cfg.loss.tau)
    else:
        hc = None

    # net_s = nn.Sequential(MLP([X.shape[1], cfg.model.hid]), nn.Linear(cfg.model.hid, data["num_classes"]))
    net_s = MyMLPs(X.shape[1], cfg.model.hid, data["num_classes"])
    optimizer = optim.Adam(net_s.parameters(), lr=0.01, weight_decay=5e-4)
    net_s = net_s.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train_stu(net_s, X, G, lbl, out_t, train_mask, optimizer, hc=hc, lamb=cfg.loss.lamb)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = valid_stu(net_s, X, lbl, val_mask, evaluator)
            if val_res > best_val:
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net_s.state_dict())
    # test
    net_s.load_state_dict(best_state)
    res_s = test_stu(net_s, X, lbl, test_mask, evaluator, cfg.data.ft_noise_level)
    logging.info(f"student test best epoch: {best_epoch}, res: {res_s}\n")
    # compute topology score
    emb_t = net(X, G, get_emb=True).detach()
    emb_s = net_s(X, get_emb=True).detach()
    tos_t = ho_topology_score(emb_t, G)
    tos_s = ho_topology_score(emb_s, G)
    logging.info(f"teacher topology score: {tos_t}")
    logging.info(f"student topology score: {tos_s}\n")
    return {"t": res_t, "s": res_s}


@hydra.main(config_path=".", config_name="trans_config", version_base="1.1")
def main(cfg: DictConfig):
    res = exp(2023, cfg)
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(f"teacher: {res['t']}")
    logging.info(f"student: {res['s']}")


if __name__ == "__main__":
    main()
