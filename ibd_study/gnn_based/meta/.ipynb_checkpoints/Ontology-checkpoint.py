class Ontology:
    def __init__(self, ontology_name):
        self.ontology_name = ontology_name
        self.ontology_layers = []

    def add_layer(self, layer):
        self.ontology_layers.append(layer)

    def add_layers(self, layers):
        for layer in layers:
            self.add_layer(layer)

    def __repr__(self):
        repr_out = f"Ontology: {self.ontology_name} containing {len(self.ontology_layers)} layers:\n"
        for layer in self.ontology_layers:
            repr_out += f"\t{layer}\n"
        return repr_out

    def __len__(self):
        return len(self.ontology_layers)
