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
    def __init__(self, ontology_list, num_classes, hidden_dim = 1, dropout=0.0, num_layers = None, protein_embeddings = None):
        super(GNN, self).__init__()
        self.module_list = []
        self.ontology_list = ontology_list
        ontology:Ontology = ontology_list[0]

        self.ontology_layer_params_dict = torch.nn.ModuleDict()
        self.ontology_layer_scaler_dict = torch.nn.ModuleDict()
        self.scatter_fun_dict = dict()
        self.num_layers = num_layers if num_layers else len(ontology.ontology_layers)
        for i in range(self.num_layers):
            ontology_layer:OntologyLayer = ontology.ontology_layers[i]
            in_dim = 1 #int(ontology_layer.num_inputs)
            if i != 0: in_dim = hidden_dim
            pre_scatter_params = torch.nn.Linear(in_dim, hidden_dim, bias=False)
            num_scattered_features = ontology_layer.edge_index[1].max()+1
            scaler = RunningMinMaxScaler(num_features=num_scattered_features*hidden_dim)
            self.ontology_layer_params_dict[ontology_layer.ontology_layer_name] = pre_scatter_params
            self.ontology_layer_scaler_dict[ontology_layer.ontology_layer_name] = scaler
            ontology_scatter = OntologyScatter(ontology_layer=ontology_layer)
            self.scatter_fun_dict[ontology_layer.ontology_layer_name] = ontology_scatter.scatter
        self.out = Linear((ontology_layer.edge_index[1].max() +1)*hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout)
        #self.weight_init() ## Zero init
        self.is_debug = False #True

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        # for param_dict in self.ontology_layer_params_dict.values():
        #     torch.nn.init.xavier_normal_(param_dict.unsqueeze(0))

    def forward(self, data):
        ontology = self.ontology_list[0]
        for i in range(self.num_layers):
            if self.is_debug: print(20*"_")
            ontology_layer:OntologyLayer = ontology.ontology_layers[i]
            if i == 0:
                data = data.unsqueeze(2)
            if self.is_debug: print(data.shape)
            transformed_features =  self.ontology_layer_params_dict[ontology_layer.ontology_layer_name](data) ## Hadamard Product to make it trainable
            if self.is_debug: print(transformed_features.shape)
            aggreg_feature_view = self.scatter_fun_dict[ontology_layer.ontology_layer_name](transformed_features) ## Aggregation for knowledge-driven feature reduction
            #aggreg_feature_view = torch.flatten(aggreg_feature_view, start_dim=1)
            if self.is_debug: print(aggreg_feature_view.shape)
            init_shape = aggreg_feature_view.shape
            aggreg_feature_view = aggreg_feature_view.flatten(start_dim=1)
            aggreg_feature_view = self.ontology_layer_scaler_dict[ontology_layer.ontology_layer_name](aggreg_feature_view) ## Scaling since network is imbalanced and we sum
            aggreg_feature_view = aggreg_feature_view.reshape(init_shape)
            aggreg_feature_view = self.dropout(aggreg_feature_view) ## Dropout
            data = torch.relu(aggreg_feature_view) ## Activation fun
            if self.is_debug: print(data.shape)
            if self.is_debug: print(20 * "_")
        if self.is_debug: print(data.shape)
        data = data.flatten(start_dim=1)
        if self.is_debug:
            print(data.shape)

        out = self.out(data) ## Final SLP/LR
        if self.is_debug: print(out.shape)
        if self.is_debug: raise Exception("Debug mode")
        return out.squeeze(-1)