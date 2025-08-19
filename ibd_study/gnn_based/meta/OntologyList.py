class OntologyList:
    def __init__(self, project_name):
        self.project_name = project_name
        self.ontology_list = []

    def add_ontology(self, ontology):
        self.ontology_list.append(ontology)

    def __repr__(self):
        repr_out = "Project Name: " + self.project_name + "\n"
        for layer in self.ontology_list:
            repr_out += f"\t{layer}\n"
        return repr_out