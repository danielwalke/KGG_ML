import torch
from torch import nn, Tensor
from torch.nn import Linear
from torch_geometric.utils import scatter

from ibd_study.gnn_based.architecture.Scaler import RunningMinMaxScaler
from ibd_study.gnn_based.meta.Ontology import Ontology
from ibd_study.gnn_based.meta.OntologyLayer import OntologyLayer
from ibd_study.gnn_based.meta.OntologyList import OntologyList

class OntologyScatter:
    def __init__(self, ontology_layer: OntologyLayer):
        self.ontology_layer = ontology_layer

    def scatter(self, features: Tensor) -> Tensor:
        lifted_feature_view = torch.index_select(features, 1, self.ontology_layer.edge_index[0])
        aggreg_feature_view = scatter(lifted_feature_view, self.ontology_layer.edge_index[1], dim=1, reduce="mean") #sum
        return aggreg_feature_view

class GNN(torch.nn.Module):
    def __init__(self, ontology_list:OntologyList, num_classes, dropout=0.0):
        super(GNN, self).__init__()
        self.module_list = []
        self.ontology_list = ontology_list
        ontology:Ontology = ontology_list[0]
        self.ontology_layer_params_dict = torch.nn.ParameterDict()
        self.ontology_layer_scaler_dict = torch.nn.ModuleDict()
        self.scatter_fun_dict = dict()
        for i in range(len(ontology.ontology_layers)):
            ontology_layer:OntologyLayer = ontology.ontology_layers[i]
            pre_scatter_params = torch.nn.Parameter(torch.randn(ontology_layer.num_inputs), requires_grad=True) ## TODO extend to multiple hidden dims
            num_scattered_features = ontology_layer.edge_index[-1].max()+1
            scaler = RunningMinMaxScaler(num_features=num_scattered_features)
            self.ontology_layer_params_dict[ontology_layer.ontology_layer_name] = pre_scatter_params
            self.ontology_layer_scaler_dict[ontology_layer.ontology_layer_name] = scaler
            ontology_scatter = OntologyScatter(ontology_layer=ontology_layer)
            self.scatter_fun_dict[ontology_layer.ontology_layer_name] = ontology_scatter.scatter
        self.out = Linear(ontology_layer.edge_index[-1].max() +1, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.weight_init() ## Zero init

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        for param_dict in self.ontology_layer_params_dict.values():
            torch.nn.init.xavier_normal_(param_dict.unsqueeze(0))

    def forward(self, data):
        ontology = self.ontology_list[0]
        for ontology_layer in ontology.ontology_layers:
            transformed_features =  data * self.ontology_layer_params_dict[ontology_layer.ontology_layer_name] ## Hadamard Product to make it trainable
            aggreg_feature_view = self.scatter_fun_dict[ontology_layer.ontology_layer_name](transformed_features) ## Aggregation for knowledge-driven feature reduction
            #aggreg_feature_view = self.ontology_layer_scaler_dict[ontology_layer.ontology_layer_name](aggreg_feature_view) ## Scaling since network is imbalanced and we sum
            aggreg_feature_view = self.dropout(aggreg_feature_view) ## Dropout
            data = torch.relu(aggreg_feature_view) ## Activation fun
        out = self.out(data) ## Final SLP/LR
        return out.squeeze(-1)