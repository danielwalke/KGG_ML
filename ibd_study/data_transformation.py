import pandas as pd
import numpy as np


# def get_embeddings(word_groups: list[str]) -> np.ndarray:
#     from sentence_transformers import SentenceTransformer
#     embedding_model = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb")
#     embeddings = embedding_model.encode(word_groups, show_progress_bar=False)
#     return np.vstack(embeddings)

def write_tax_edges(data_df):
    tax_columns = []
    for tax_level in ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]:
        tax_columns.append(f"task_1::Taxonomic_Annotation_Task_1::{tax_level}")
    tax_edges = data_df[tax_columns].copy()
    for col in tax_columns:
        tax_edges[col.split("::")[-1]] = tax_edges[col].astype("category").cat.codes
        tax_edges.pop(col)
    tax_edges = tax_edges.reset_index()
    tax_edges.to_csv("data/tax_edges.csv", index=False)

def parse_and_transform_column_in_exploded_categorical(data_df, column, rm_nan = False):
    data_df = data_df.copy()
    data_df[f"{column}_transformed"] = data_df[column].str.strip('{}').str.split(', ').apply(
        lambda lst: [item.strip("'") for item in lst if item.strip("'") != '-'])
    data_df = data_df.explode(f"{column}_transformed")
    if rm_nan:
        data_df.dropna(subset=[f"{column}_transformed"], inplace=True)
        return data_df, None
    empty_mask = data_df[f"{column}_transformed"].isna()
    data_df.loc[empty_mask, f"{column}_transformed"] = np.arange(empty_mask.sum())
    return data_df, empty_mask

def write_function_edges(data_df):
    data_df_ko, empty_mask = parse_and_transform_column_in_exploded_categorical(data_df, "KO")
    function_edges = pd.DataFrame()
    function_edges["index"] = data_df_ko.index
    function_edges["trg"] = data_df_ko["KO_transformed"].astype("category").cat.codes.values
    function_edges.to_csv("data/function_edges.csv", index=False)

    data_df_go, _ = parse_and_transform_column_in_exploded_categorical(data_df_ko[empty_mask], "GO:Term", True)
    function_edges = pd.DataFrame()
    function_edges["index"] = data_df_go.index
    function_edges["trg"] = data_df_go["GO:Term_transformed"].astype("category").cat.codes.values
    function_edges.to_csv("data/function_edges_go.csv", index=False)




data_df = pd.read_csv("data/merged_df.csv", sep=";")
annotation_df = pd.read_csv("data/SampleAnnotation.csv", sep=";", index_col=0)

annotation_label = "disease" # "disease" # "condition"
conditions = annotation_df.loc[annotation_label].reset_index()

## Embedding
# protein_str_repr = data_df.loc[:, ['task_0::Functional_Annotation_Task_1::desc', 'task_0::Functional_Annotation_Task_1::subrole', 'EC']].agg(' '.join, axis=1)
# embeddings = get_embeddings(protein_str_repr.tolist(), embedding_model)
# np.save('embeddings.npy', embeddings)

write_tax_edges(data_df)
write_function_edges(data_df)

transposed_data_df = data_df.transpose().reset_index()

merged_data_df = pd.merge(transposed_data_df, conditions, on="index")
if annotation_label == "disease":
    print(merged_data_df[annotation_label].value_counts())
    ## Merged remission and active
    merged_data_df[annotation_label] = merged_data_df[annotation_label].str.replace("UCa", "UC").replace("UCr", "UC")
print(merged_data_df[annotation_label].value_counts())
merged_data_df["condition"] = merged_data_df[annotation_label].astype("category").cat.codes

if annotation_label != "condition":
    merged_data_df.pop(annotation_label)

merged_data_df.to_csv("data/transformed_df.csv", index=False)
