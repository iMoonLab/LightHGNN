import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import hydra
import logging
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from utils import load_data, product_split, sub_hypergraph, fix_iso_v

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dhg.nn import MLP
from dhg import Hypergraph
from dhg.random import set_seed
from dhg.utils import split_by_num
from dhg.models import HGNNP, HGNN, HNHN, UniGCN
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
def test(net, X_t, G_t, lbls_t, mask_t, X, G, lbls, mask, prod_mask, evaluator):
    net.eval()
    # transductive
    outs_t = net(X_t, G_t)
    res_t = evaluator.test(lbls_t[mask_t], outs_t[mask_t])
    # # inductive
    outs = net(X, G)
    res_i = evaluator.test(lbls[mask], outs[mask])
    # product
    outs = net(X, G)
    res_p = evaluator.test(lbls[prod_mask], outs[prod_mask])
    res = {}
    for k, v in res_p.items():
        res[f"prod_{k}"] = v
    for k, v in res_i.items():
        res[f"ind_{k}"] = v
    for k, v in res_t.items():
        res[f"trans_{k}"] = v
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
def test_stu(net, X_t, lbls_t, mask_t, X, lbls, mask, prod_mask, evaluator):
    net.eval()
    # transductive
    outs_t = net(X_t)
    res_t = evaluator.test(lbls_t[mask_t], outs_t[mask_t])
    # inductive
    outs = net(X)
    res_i = evaluator.test(lbls[mask], outs[mask])
    # product
    outs = net(X)
    res_p = evaluator.test(lbls[prod_mask], outs[prod_mask])
    res = {}
    for k, v in res_p.items():
        res[f"prod_{k}"] = v
    for k, v in res_i.items():
        res[f"ind_{k}"] = v
    for k, v in res_t.items():
        res[f"trans_{k}"] = v
    return res


# =========================================================================
def exp(seed, cfg: DictConfig):
    set_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    data, edge_list = load_data(cfg.data.name)

    G = Hypergraph(data["num_vertices"], edge_list)
    G = fix_iso_v(G)
    train_mask, val_mask, test_mask = split_by_num(
        data["num_vertices"], data["labels"], cfg.data.num_train, cfg.data.num_val
    )
    obs_idx, obs_train_mask, obs_val_mask, obs_test_mask, test_ind_mask = product_split(
        train_mask, val_mask, test_mask, cfg.data.test_ind_ratio
    )
    G_t = sub_hypergraph(G, obs_idx)
    G_t = fix_iso_v(G_t)
    X, lbl = data["features"], data["labels"]
    X_t, lbl_t = X[obs_idx], lbl[obs_idx]

    if cfg.model.teacher == "hgnn":
        net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.teacher == "hgnnp":
        net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.teacher == "hnhn":
        net = HNHN(X.shape[1], 32, data["num_classes"], use_bn=False)
    elif cfg.model.teacher == "unigcn":
        net = UniGCN(X.shape[1], 32, data["num_classes"], use_bn=False)
    else:
        raise NotImplementedError

    # train teacher
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    X, lbl, G = X.to(device), lbl.to(device), G.to(device)
    X_t, lbl_t, G_t = X_t.to(device), lbl_t.to(device), G_t.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X_t, G_t, lbl_t, obs_train_mask, optimizer)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = valid(net, X_t, G_t, lbl_t, obs_val_mask, evaluator)
            if val_res > best_val:
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    # test
    net.load_state_dict(best_state)
    res_t = test(net, X_t, G_t, lbl_t, obs_test_mask, X, G, lbl, test_ind_mask, test_mask, evaluator)
    logging.info(f"teacher test best epoch: {best_epoch}, res: {res_t}")

    # -------------------------------------------------------------------------------------
    # train student
    out_t = net(X_t, G_t).detach()
    if cfg.model.student == "light_hgnnp":
        hc = HighOrderConstraint(net, X_t, G_t, noise_level=cfg.data.hc_noise_level, tau=cfg.loss.tau)
    else:
        hc = None

    if cfg.data.ft_noise_level > 0:
        X = (1 - cfg.data.ft_noise_level) * X + cfg.data.ft_noise_level * torch.randn_like(X)

    net_s = nn.Sequential(MLP([X.shape[1], cfg.model.hid]), nn.Linear(cfg.model.hid, data["num_classes"]))
    optimizer = optim.Adam(net_s.parameters(), lr=0.01, weight_decay=5e-4)
    net_s = net_s.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train_stu(net_s, X_t, G_t, lbl_t, out_t, obs_train_mask, optimizer, hc=hc, lamb=cfg.loss.lamb)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = valid_stu(net_s, X_t, lbl_t, obs_val_mask, evaluator)
            if val_res > best_val:
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net_s.state_dict())
    # test
    net_s.load_state_dict(best_state)
    res_s = test_stu(net_s, X_t, lbl_t, obs_test_mask, X, lbl, test_ind_mask, test_mask, evaluator)
    logging.info(f"student test best epoch: {best_epoch}, res: {res_s}\n")
    return {"t": res_t, "s": res_s}


@hydra.main(config_path=".", config_name="prod_config", version_base="1.1")
def main(cfg: DictConfig):
    res = exp(2023, cfg)
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(f"teacher: {res['t']}")
    logging.info(f"student: {res['s']}")


if __name__ == "__main__":
    main()
