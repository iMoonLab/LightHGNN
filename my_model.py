import torch
import torch.nn as nn

import dhg
from dhg.nn import MLP
from dhg.nn import GCNConv
from dhg.nn import HGNNConv


class MyGCN(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.layers0 = GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.layers1 = GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)

    def forward(self, X: torch.Tensor, g: "dhg.Graph", get_emb=False) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """        
        emb = self.layers0(X, g)
        X = self.layers1(emb, g)
        if get_emb:
            return emb
        else:
            return X


class MyHGNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers0 = HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.layers1 = HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph", get_emb=False) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        emb = self.layers0(X, hg)
        X = self.layers1(emb, hg)
        if get_emb:
            return emb
        else:
            return X


class MyMLPs(nn.Module):
    def __init__(self, dim_in, dim_hid, n_classes) -> None:
        super().__init__()
        self.layer0 = MLP([dim_in, dim_hid])
        self.layer1 = nn.Linear(dim_hid, n_classes)
    
    def forward(self, X, get_emb=False):
        emb = self.layer0(X)
        X = self.layer1(emb)
        if get_emb:
            return emb
        else:
            return X
