class OntologyLayer:
    def __init__(self, src_type, trg_type, num_inputs, edge_index):
        self.src_type = src_type
        self.trg_type = trg_type
        self.num_inputs = num_inputs
        self.edge_index = edge_index
        self.ontology_layer_name = f"{self.src_type}-{self.trg_type}"

    def __repr__(self):
        return f"Layer from {self.src_type} ({self.num_inputs}) to {self.trg_type} - Shape: {self.edge_index.shape}"
