from torch import nn
from ibd_study.gnn_based.meta.Ontology import  Ontology
from ibd_study.gnn_based.architecture.GNNOntologylayer import GNNOntologyLayer

class GNNOntology(nn.Module):
    def __init__(self, ontology:Ontology, hidden_dim, num_heads, act_function, dropout):
        super(GNNOntology, self).__init__()
        self.ontology:Ontology = ontology
        self.dropout = nn.Dropout(dropout)
        self.act_function = act_function
        self.gnn_list = nn.ModuleList()
        for i, ontology_layer in enumerate(self.ontology.ontology_layers):
            in_dim = 1 if i == 0 else self.gnn_list[i-1].out_dim
            gnn_layer = GNNOntologyLayer(ontology_layer, in_dim, hidden_dim, num_heads, dropout)
            self.gnn_list.append(gnn_layer)
        num_scattered_out = self.ontology.ontology_layers[-1].edge_index[-1].max()+1
        self.final_out_dim = self.gnn_list[-1].out_dim * num_scattered_out

    def forward(self, x):
        for i, ontology_layer in enumerate(self.ontology.ontology_layers):
            gnn_layer = self.gnn_list[i]
            x = gnn_layer(x)
            x = self.dropout(x)
            x = self.act_function(x)
        return x
