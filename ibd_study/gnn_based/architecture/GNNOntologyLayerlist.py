import torch
from numpy.ma.core import concatenate
from torch import nn, Tensor
from ibd_study.gnn_based.architecture.GNNOntology import GNNOntology


def create_classifier(in_dim, hidden_dim, num_classes, num_layers, dropout=0.5):
    layers = []
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)

class GNN(nn.Module):
    def __init__(self, ontology_list:list, hidden_dim, num_heads, act_function, dropout, num_layers_clf, num_classes):
        super(GNN, self).__init__()
        self.ontology_list = ontology_list
        self.ontology_module_list = nn.ModuleList()
        self.summed_final_out_dim = 0
        for ontology in ontology_list:
            gnn_ontology_module = GNNOntology(ontology, hidden_dim, num_heads, act_function, dropout)
            self.ontology_module_list.append(gnn_ontology_module)
            self.summed_final_out_dim += gnn_ontology_module.final_out_dim.item()
        self.norm = nn.LayerNorm(self.summed_final_out_dim)
        self.out_proj = create_classifier(self.summed_final_out_dim, hidden_dim, num_classes, num_layers_clf, dropout)



    def forward(self, x:Tensor) -> Tensor:
        transformed_x_layers = []
        for gnn_ontology_module in self.ontology_module_list:
            ontology_transformed_x:Tensor = gnn_ontology_module(x)
            transformed_x_layers.append(ontology_transformed_x.flatten(start_dim=1))
        concatenated_transformed_x = torch.cat(transformed_x_layers, dim=1)
        concatenated_transformed_x = self.norm(concatenated_transformed_x)
        return self.out_proj(concatenated_transformed_x)
