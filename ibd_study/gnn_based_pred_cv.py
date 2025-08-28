import pandas as pd
import numpy as np
import torch
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe, space_eval
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
from tqdm import tqdm
import datetime
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from ibd_study.gnn_based.architecture.GNNModel import GNNModel
from ibd_study.gnn_based.meta.Ontology import Ontology
from ibd_study.gnn_based.meta.OntologyLayer import OntologyLayer
from ibd_study.gnn_based.seed.SeedSetter import set_all_seeds
from ibd_study.logger import logging

NESTED_CV = (10, 5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

here = os.path.dirname(__file__)
data_df = pd.read_csv(f"{here}/data/transformed_df.csv")
sample_names = data_df.pop("index")
y = data_df.pop("condition").values
X = data_df.values
print(np.unique(y, return_counts=True))

proteins_over_samples_df = data_df.transpose().reset_index()
proteins_over_samples_df["index"] = proteins_over_samples_df["index"].astype(np.int64)


ko_edges = pd.read_csv(f"{here}/data/function_edges.csv")
go_edges = pd.read_csv(f"{here}/data/function_edges_go.csv")
ko_edges["index"] = ko_edges["index"].astype(np.int64)
go_edges["index"] = go_edges["index"].astype(np.int64)

ko_edge_index = torch.from_numpy(ko_edges.loc[:, ["index", "trg"]].values.transpose()).type(torch.long).to(device)
go_edge_index = torch.from_numpy(go_edges.loc[:, ["index", "trg"]].values.transpose()).type(torch.long).to(device)

ontology_layer_ko = OntologyLayer("protein", "ko", data_df.values.shape[-1], ko_edge_index)
ontology_layer_go = OntologyLayer("protein", "go", data_df.values.shape[-1], go_edge_index)
ko_ontology = Ontology("KO")
ko_ontology.add_layer(ontology_layer_ko)

go_ontology = Ontology("GO")
go_ontology.add_layer(ontology_layer_go)
ontology_list = [ko_ontology,go_ontology]


outer_cv = KFold(n_splits=NESTED_CV[0], shuffle=True, random_state=42)
fold_accuracies = []
total_cm = None
class_labels = np.unique(y)

print("\nStarting Nested Cross-Validation for GNN...")

for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X), 1):
    print(f"\n--- Processing Outer Fold {fold_idx}/{NESTED_CV[0]} ---")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

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
        "num_heads": hp.choice("num_heads", [1, 2, 4]),
        "num_layers": hp.quniform("num_layers", 1, 3, q=1)
    }

    def get_best_params(X_train, y_train):
        def objective(params):
            clf = GNNModel(ontology_list, num_classes, params["lr"], params["weight_decay"], 500, params["dropout"], device=device, hidden_dim = params["hidden_dim"], num_layers =  params["num_layers"], num_heads=params["num_heads"])
            skf = KFold(n_splits=NESTED_CV[1], shuffle=True, random_state=42)
            accuracy_scores = []

            for i, (train_index, val_index) in enumerate(skf.split(X_train.cpu())):
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
            rstate=np.random.default_rng(42),
            trials=trials
        )

        final_best_params = space_eval(space, best_params_raw)
        return final_best_params

    best_params = get_best_params(X_train, y_train)
    print(best_params) ##Check mean aggr
    model = GNNModel(ontology_list, num_classes, best_params["lr"], best_params["weight_decay"], 500, best_params["dropout"], device=device, hidden_dim = best_params["hidden_dim"], num_layers =  best_params["num_layers"], num_heads=best_params["num_heads"])
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    acc = metrics.accuracy_score(y_test.cpu().numpy(), y_pred_test)

    print(f"Accuracy: {acc}")
    fold_accuracies.append(acc)
    log_message = (
        f"Model Run Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  File: {os.path.basename(__file__)} (Fold: {fold_idx} / {NESTED_CV[0]})\n"
        f"  Accuracy: {acc:.4f}\n"
        f" Best params: {best_params}\n"
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

 ## Ideas for improvements: Aggr as hyperparam; Knowledge based on protein similarity attached as information (name and taxa embeddings)

