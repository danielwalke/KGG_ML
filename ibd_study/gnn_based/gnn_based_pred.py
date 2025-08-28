import pandas as pd
import numpy as np
import torch
import datetime
import os
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe, space_eval
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
from tqdm import tqdm

from ibd_study.gnn_based.architecture.GNNModel import GNNModel
from ibd_study.gnn_based.meta.Ontology import Ontology
from ibd_study.gnn_based.meta.OntologyLayer import OntologyLayer
from ibd_study.gnn_based.seed.SeedSetter import set_all_seeds
from ibd_study.logger import logging
import torch
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)#5833
seed = 42
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
g = torch.Generator()
g.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

data_df = pd.read_csv("../data/transformed_df.csv")
sample_names = data_df.pop("index")
y = data_df.pop("condition").values
print(np.unique(y, return_counts=True))

proteins_over_samples_df = data_df.transpose().reset_index()
proteins_over_samples_df["index"] = proteins_over_samples_df["index"].astype(np.int64)
tax_edges = pd.read_csv("../data/tax_edges.csv")

X = data_df.values

ko_edges = pd.read_csv("../data/function_edges.csv")
go_edges = pd.read_csv("../data/function_edges_go.csv")
ko_edges["index"] = ko_edges["index"].astype(np.int64)
go_edges["index"] = go_edges["index"].astype(np.int64)

ko_edge_index = torch.from_numpy(ko_edges.loc[:, ["index", "trg"]].values.transpose()).type(torch.long).to(device)
go_edge_index = torch.from_numpy(go_edges.loc[:, ["index", "trg"]].values.transpose()).type(torch.long).to(device)

ontology_layer_ko = OntologyLayer("protein", "ko", data_df.values.shape[-1], ko_edge_index)
ontology_layer_go = OntologyLayer("protein", "go", data_df.values.shape[-1], go_edge_index)

# species_edge_index = torch.from_numpy(tax_edges.loc[:, ["index", "species"]].values.transpose()).type(torch.long).to(device)
# genus_edge_index = torch.from_numpy(tax_edges.loc[:, ["species", "genus"]].values.transpose()).type(torch.long).to(device)
# family_edge_index = torch.from_numpy(tax_edges.loc[:, ["genus", "family"]].values.transpose()).type(torch.long).to(device)
# order_edge_index = torch.from_numpy(tax_edges.loc[:, ["family", "order"]].values.transpose()).type(torch.long).to(device)
# class_edge_index = torch.from_numpy(tax_edges.loc[:, ["order", "class"]].values.transpose()).type(torch.long).to(device)
# phylum_edge_index = torch.from_numpy(tax_edges.loc[:, ["class", "phylum"]].values.transpose()).type(torch.long).to(device)
# superkingdom_edge_index = torch.from_numpy(tax_edges.loc[:, ["phylum", "superkingdom"]].values.transpose()).type(torch.long).to(device)
#
# ontology_layer_species = OntologyLayer("protein", "species", data_df.values.shape[-1], species_edge_index)
# ontology_layer_genus = OntologyLayer("species", "genus", species_edge_index[-1].max()+1, genus_edge_index)
# ontology_layer_family = OntologyLayer("genus", "family", genus_edge_index[-1].max()+1, family_edge_index)
# ontology_layer_order = OntologyLayer("family", "order", family_edge_index[-1].max()+1, order_edge_index)
# ontology_layer_class = OntologyLayer("order", "class", order_edge_index[-1].max()+1, class_edge_index)
# ontology_layer_phylum = OntologyLayer("class", "phylum", class_edge_index[-1].max()+1, phylum_edge_index)
# ontology_layer_superkingdom = OntologyLayer("phylum", "superkingdom", phylum_edge_index[-1].max()+1, superkingdom_edge_index)
#
# tax_ontology = Ontology("Taxonomy")
# tax_ontology.add_layer(ontology_layer_species)
# tax_ontology.add_layer(ontology_layer_genus)
# tax_ontology.add_layer(ontology_layer_family)
# tax_ontology.add_layer(ontology_layer_order)
# tax_ontology.add_layer(ontology_layer_class)
# tax_ontology.add_layer(ontology_layer_phylum)
# tax_ontology.add_layer(ontology_layer_superkingdom)
# ##Average Accuracy: 0.6333 ± 0.0425 -> no early stopping tested or scaling of initial features ## TODO
# ontology_list = [tax_ontology]
ko_ontology = Ontology("KO")
ko_ontology.add_layer(ontology_layer_ko)

go_ontology = Ontology("GO")
go_ontology.add_layer(ontology_layer_go)
ontology_list = [ko_ontology,go_ontology]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train).to(device).type(torch.float)
X_test = torch.from_numpy(X_test).to(device).type(torch.float)
y_train = torch.from_numpy(y_train).to(device).type(torch.long)
y_test = torch.from_numpy(y_test).to(device).type(torch.long)

num_classes = torch.unique(y_train).shape[0]

space = {
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-1)),
    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-1)),
    'dropout': hp.uniform('dropout', 0, 8e-1),
    "hidden_dim": hp.quniform("hidden_dim", 8, 256, q=4),
    "num_heads": hp.choice("num_heads", [1,2, 4]),
    "num_layers": hp.quniform("num_layers", 1, 3, q=1)
}

def get_best_params(X_train, y_train):
    """
    Objective function for Hyperopt to find the best hyperparameters
    for Logistic Regression using stratified cross-validation.
    """

    def objective(params):
        clf = GNNModel(ontology_list, num_classes, params["lr"], params["weight_decay"], 1000, params["dropout"],   hidden_dim = params["hidden_dim"], num_layers =  params["num_layers"], num_heads = params["num_heads"], device=device)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        accuracy_scores = []

        for i, (train_index, val_index) in enumerate(skf.split(X_train.cpu(), y_train.cpu())):
            X_train_inner, X_val = X_train[train_index], X_train[val_index]
            y_train_inner, y_val = y_train[train_index], y_train[val_index]

            clf.fit(X_train_inner, y_train_inner)

            score = metrics.accuracy_score(y_val.cpu().numpy(), clf.predict(X_val))
            accuracy_scores.append(score)

        mean_accuracy = np.mean(accuracy_scores)
        return {'loss': -mean_accuracy, 'status': STATUS_OK}

    trials = Trials()
    best_params_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    final_best_params = space_eval(space, best_params_raw)
    return final_best_params

best_params = get_best_params(X_train, y_train)
print(best_params)
# best_params = {
#     "weight_decay": 1e-5,
#     "lr": 1e-3,
#     "dropout": 0.5,
#     "hidden_dim": 512,
#     "num_layers": 1,
#     "num_heads": 16
# }

acc_scores = []
for i in tqdm(range(1)):
    set_all_seeds(i)
    model = GNNModel(ontology_list, num_classes, best_params["lr"], best_params["weight_decay"], 1000, best_params["dropout"], device=device, hidden_dim = best_params["hidden_dim"], num_layers =   best_params["num_layers"], num_heads= best_params["num_heads"])
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    acc_scores.append(metrics.accuracy_score(y_test.cpu().numpy(), y_pred_test))
print(f"Accuracy: {np.mean(acc_scores)} +- {np.std(acc_scores)}")

class_labels = np.unique(y)
cm = metrics.confusion_matrix(y_test.cpu().numpy(), y_pred_test, labels=class_labels)
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
print(df_cm)

log_message = (
    f"Model Run Results with layers - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"  File: {os.path.basename(__file__)}\n"
    f"  Accuracy: {np.mean(acc_scores):.2f} +- {np.std(acc_scores)}\n"
    f"  Confusion Matrix: {df_cm}\n"
    f"--------------------------------------------------"
)
logging.info(log_message)

