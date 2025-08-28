import torch
import math
from torch import nn, Tensor
from torch_geometric.utils import scatter, softmax
from ibd_study.gnn_based.meta.OntologyLayer import OntologyLayer
import torch.nn.functional as F

class GNNOntologyLayer(nn.Module):
    def __init__(self, ontology_layer:OntologyLayer, in_dim, model_dim, num_heads, dropout, scatter_fun = "mean"):
        super(GNNOntologyLayer, self).__init__()
        self.ontology_layer = ontology_layer

        assert model_dim % num_heads == 0, "Model dimension must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.proj = nn.Linear(in_dim, model_dim, bias=False)
        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)

        self.scatter_fun = scatter_fun
        self.norm = nn.LayerNorm(model_dim)
        self.out_dim = model_dim #self.ontology_layer.edge_index[-1].max() + 1 #model_dim * (self.ontology_layer.edge_index[-1].max() + 1)
        self.dropout = torch.nn.Dropout(dropout)

    def lift_and_scatter(self, features: Tensor) -> Tensor:
        lifted_feature_view = torch.index_select(features, 1, self.ontology_layer.edge_index[0])
        aggreg_feature_view = scatter(lifted_feature_view, self.ontology_layer.edge_index[1], dim=1, reduce=self.scatter_fun)
        return aggreg_feature_view

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.proj(x) # NxOxH
        x = self.dropout(x)
        lifted_x = torch.index_select(x, 1, self.ontology_layer.edge_index[0])
        Q = self.W_q(lifted_x)
        K = self.W_k(lifted_x)
        V = self.W_v(lifted_x)
        alpha = (Q * K).sum(dim=-1) * 1 / math.sqrt(self.head_dim)
        attention_weights = softmax(src=alpha, index=self.ontology_layer.edge_index[1], dim=-1)
        weighted_V = V * attention_weights.unsqueeze(-1)
        aggregated_embeddings = scatter(
            src=weighted_V,
            index=self.ontology_layer.edge_index[1],
            dim=1,
            dim_size=self.ontology_layer.edge_index[1].max()+1,
            reduce='sum'
        )
        skip_connections = scatter(lifted_x, self.ontology_layer.edge_index[1], dim=1, dim_size=self.ontology_layer.edge_index[1].max()+1,
                                  reduce='mean')
        aggregated_embeddings = aggregated_embeddings + skip_connections
        aggregated_embeddings = self.norm(aggregated_embeddings)
        return aggregated_embeddings
