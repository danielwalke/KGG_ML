import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def get_embeddings(word_groups: list[str], model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(word_groups, show_progress_bar=False)
    return embeddings

data_df = pd.read_csv("data/merged_df.csv", sep=";")
annotation_df = pd.read_csv("data/SampleAnnotation.csv", sep=";", index_col=0)
embedding_model = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb")
annotation_label = "disease" # "disease" # "condition"
conditions = annotation_df.loc[annotation_label].reset_index()

protein_str_repr = data_df.loc[:, ['task_0::Functional_Annotation_Task_1::desc', 'task_0::Functional_Annotation_Task_1::og', 'task_0::Functional_Annotation_Task_1::main role', 'task_0::Functional_Annotation_Task_1::subrole']].agg(' '.join, axis=1)
embeddings = get_embeddings(protein_str_repr.tolist(), embedding_model)
embedding_stack = np.vstack(embeddings)
np.save('embeddings.npy', embedding_stack)

tax_columns = []
for tax_level in ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]:
    tax_columns.append(f"task_1::Taxonomic_Annotation_Task_1::{tax_level}")
tax_edges = data_df[tax_columns].copy()
for col in tax_columns:
    tax_edges[col.split("::")[-1]] = tax_edges[col].astype("category").cat.codes
    tax_edges.pop(col)
tax_edges = tax_edges.reset_index()
tax_edges.to_csv("data/tax_edges.csv", index=False)

transposed_data_df = data_df.transpose().reset_index()

# labels_in_annotation_df = set(conditions["index"].values.tolist())
# labels_in_data_df = set(transposed_data_df["index"].values.tolist())
# print(labels_in_data_df ^labels_in_annotation_df)
# print(labels_in_data_df.intersection(labels_in_annotation_df))
merged_data_df = pd.merge(transposed_data_df, conditions, on="index")
if annotation_label == "disease":
    ## Merged remission and active
    merged_data_df[annotation_label] = merged_data_df[annotation_label].str.replace("UCa", "UC").replace("UCr", "UC")
print(merged_data_df[annotation_label].value_counts())
merged_data_df["condition"] = merged_data_df[annotation_label].astype("category").cat.codes

if annotation_label != "condition":
    merged_data_df.pop(annotation_label)

merged_data_df.to_csv("data/transformed_df.csv", index=False)
## TODO construct edge index
