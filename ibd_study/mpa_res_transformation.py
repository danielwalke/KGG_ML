import pandas as pd
import numpy as np


data_df = pd.read_csv("data/mpapr_res.csv", sep="\t")
annotation_df = pd.read_csv("data/SampleAnnotation.csv", sep=";", index_col=0)

annotation_label = "disease" # "disease" # "condition"
conditions = annotation_df.loc[annotation_label].reset_index()

data_df = data_df[data_df["identified reactions"] != 0]
for col in ["index","pathway","superkingdom","kingdom","phylum","class","order","family","genus","species","identified reactions","total reactions in pathway"]:
    data_df.pop(col)
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

merged_data_df.to_csv("data/transformed_mpa_res.csv", index=False)
