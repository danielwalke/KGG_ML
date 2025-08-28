import pandas as pd
import numpy as np

def write_function_edges(data_df):
    data_df = data_df.copy()
    data_df["KO_transformed"] = data_df["KO"].str.strip('{}').str.split(', ').apply(
        lambda lst: [item.strip("'") for item in lst if item.strip("'") != '-'])
    data_df = data_df.explode("KO_transformed")
    empty_ko_mask = data_df["KO_transformed"].isna()
    data_df.loc[empty_ko_mask, "KO_transformed"] = np.arange(empty_ko_mask.sum())
    function_edges = pd.DataFrame()
    function_edges["index"] = data_df.index
    function_edges["trg"] = data_df["KO_transformed"].astype("category").cat.codes.values
    function_edges.to_csv("data/function_edges.csv", index=False)

data_df = pd.read_csv("data/merged_df.csv", sep=";")
annotation_df = pd.read_csv("data/SampleAnnotation.csv", sep=";", index_col=0)

label_dict = dict(zip(annotation_df.columns, annotation_df.loc["disease", :].str.replace("UCa", "UC").replace("UCr", "UC")))


out_df = pd.DataFrame()
out_df["Id"] = data_df.iloc[:, 0]
print(data_df.loc[:, "KO"])
out_df["KOs"] = data_df["KO"].str.strip('{}').str.replace("ko:", "").str.split(', ').apply(
        lambda lst: [item.strip("'") for item in lst if item.strip("'") != '-']).str.join(';')
out_df["ECs"] = data_df.loc[:, "EC"].str.strip('{}').str.replace("ec:", "").str.split(', ').apply(
        lambda lst: [item.strip("'") for item in lst if item.strip("'") != '-']).str.join(';')

tax_columns = []
for i, tax_level in enumerate(["superkingdom", "phylum", "class", "order", "family", "genus", "species"]):
    tax_col = f"task_1::Taxonomic_Annotation_Task_1::{tax_level}"
    out_df[tax_level] = data_df.loc[:, tax_col]
    if i == 0:
        out_df["kingdom"] = out_df.loc[:, tax_level]
out_df["description"] = pd.Series(['-' for _ in range(out_df.shape[0])])

for col in data_df.columns[27:]:
    if col not in label_dict: continue
    out_df[f"{col}"] = data_df[col]
out_df.to_csv("data/mpapt_df.csv", index=False, sep= "\t")
# Id	KOs	ECs	Superkingdom	Kingdom	Phylum	Class	Order	Family	Genus	Species	descriptions	3274_F1_1	3274_F1_2	3274_F1_3	3274_F2_1	3274_F2_2	3274_F2_3	6794_F1_1	6794_F1_2	6794_F1_3	6794_F2_1	6794_F2_2	6794_F2_3	49_F1_1	49_F1_2	49_F1_3


