import pandas as pd
import numpy as np
import torch
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe, space_eval
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
from tqdm import tqdm
import datetime
import os
from sklearn.preprocessing import StandardScaler

from ibd_study.gnn_based.architecture.GNNModel import GNNModel
from ibd_study.gnn_based.meta.Ontology import Ontology
from ibd_study.gnn_based.meta.OntologyLayer import OntologyLayer
from ibd_study.gnn_based.seed.SeedSetter import set_all_seeds
from ibd_study.logger import logging

NESTED_CV = (10, 5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_all_seeds(42)

data_df = pd.read_csv("../data/transformed_df.csv")
sample_names = data_df.pop("index")
y = data_df.pop("condition").values
print(np.unique(y, return_counts=True))

proteins_over_samples_df = data_df.transpose().reset_index()
proteins_over_samples_df["index"] = proteins_over_samples_df["index"].astype(np.int64)
tax_edges = pd.read_csv("../data/tax_edges.csv")

species_edge_index = torch.from_numpy(tax_edges.loc[:, ["index", "species"]].values.transpose()).type(torch.long).to(device)
genus_edge_index = torch.from_numpy(tax_edges.loc[:, ["species", "genus"]].values.transpose()).type(torch.long).to(device)
phylum_edge_index = torch.from_numpy(tax_edges.loc[:, ["genus", "phylum"]].values.transpose()).type(torch.long).to(device)
X = data_df.values

ontology_layer = OntologyLayer("protein", "species", data_df.values.shape[-1], species_edge_index)
ontology_layer_genus = OntologyLayer("species", "genus", species_edge_index[-1].max()+1, genus_edge_index)
ontology_layer_phylum = OntologyLayer("genus", "phylum", genus_edge_index[-1].max()+1, phylum_edge_index)

tax_ontology = Ontology("Taxonomy")
tax_ontology.add_layer(ontology_layer)
#tax_ontology.add_layer(ontology_layer_genus)
# tax_ontology.add_layer(ontology_layer_phylum)
ontology_list = [tax_ontology]


outer_cv = StratifiedKFold(n_splits=NESTED_CV[0], shuffle=True, random_state=42)
#0.6000 ± 0.1106
fold_accuracies = []
total_cm = None
class_labels = np.unique(y)

 ##TODO Check scaler here or without early stopping

print("\nStarting Nested Cross-Validation for GNN...")

for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n--- Processing Outer Fold {fold_idx}/{NESTED_CV[0]} ---")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    X_train = torch.from_numpy(X_train).to(device).type(torch.float)
    X_test = torch.from_numpy(X_test).to(device).type(torch.float)
    y_train = torch.from_numpy(y_train).to(device).type(torch.long)
    y_test = torch.from_numpy(y_test).to(device).type(torch.long)

    num_classes = torch.unique(y_train).shape[0]

    space = {
        'C': hp.loguniform('C', np.log(0.0001), np.log(100)),
        'lr': hp.loguniform('lr', np.log(1e-3), np.log(1e-1)),
        'dropout': hp.loguniform('dropout', np.log(1e-2), np.log(5e-1)),
        'epochs': hp.choice('epochs', [1000]), #hp.quniform('epochs', 50, 500, 1)
        "hidden_dim": hp.quniform("hidden_dim", 2, 64, q=2)
    }

    def get_best_params(X_train, y_train):
        def objective(params):
            clf = GNNModel(ontology_list, num_classes, params["lr"], params["C"], int(params["epochs"]), params["dropout"],  hidden_dim = params["hidden_dim"], device=device)
            skf = StratifiedKFold(n_splits=NESTED_CV[1], shuffle=True, random_state=42)
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
            trials=trials
        )

        final_best_params = space_eval(space, best_params_raw)
        return final_best_params

    best_params = get_best_params(X_train, y_train)
    # best_params = {
    #     "C": 10,
    #     "lr": 1e-2,
    #     "dropout": 0.0,
    # }

    acc_scores = []
    # for i in tqdm(range(10)):
    #     set_all_seeds(i)
    print(best_params) ##Check mean aggr
    model = GNNModel(ontology_list, num_classes, best_params["lr"], best_params["C"], int(best_params["epochs"]), best_params["dropout"], hidden_dim = best_params["hidden_dim"], device=device) # C= 10, lr = 1e-2, dropout = 0, epochs, 500
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    acc_scores.append(metrics.accuracy_score(y_test.cpu().numpy(), y_pred_test))
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    print(f"Accuracy: {mean_acc} +- {std_acc}")
    fold_accuracies.append(mean_acc)
    log_message = (
        f"Model Run Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  File: {os.path.basename(__file__)} (Fold: {fold_idx} / {NESTED_CV[0]})\n"
        f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"--------------------------------------------------"
    )
    logging.info(log_message)
print("\n--- Nested Cross-Validation Complete ---")
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"\nAverage Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
log_message = (
    f"Model Run Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"  File: {os.path.basename(__file__)}\n"
    f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n"
    f"--------------------------------------------------"
)
logging.info(log_message)

 ##TODO Make separate project -> Push on github -> Pull on a server -> Epochs as hyperparam -> Aggr as hyperparam -> Num layers as hyperparam -> More iterations -> Run there
 ## CHeck if outer loop is overfitting and if some early stopping would help
 ## If not wrking -> Extend idea with Knowledge based on protein similarity attached as infromation (name and taxa embeddings)

