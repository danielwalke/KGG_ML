import pandas as pd
import numpy as np
import torch
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe, space_eval
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
from tqdm import tqdm

from ibd_study.gnn_based.architecture.GNNModel import GNNModel
from ibd_study.gnn_based.meta.Ontology import Ontology
from ibd_study.gnn_based.meta.OntologyLayer import OntologyLayer
from ibd_study.gnn_based.seed.SeedSetter import set_all_seeds

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
X = data_df.values

ontology_layer = OntologyLayer("protein", "species", data_df.values.shape[-1], species_edge_index)
ontology_layer_genus = OntologyLayer("species", "genus", species_edge_index[-1].max()+1, genus_edge_index)

tax_ontology = Ontology("Taxonomy")
tax_ontology.add_layer(ontology_layer)
#tax_ontology.add_layer(ontology_layer_genus)
##Average Accuracy: 0.6333 ± 0.0425 -> no early stopping tested or scaling of initial features ## TODO
ontology_list = [tax_ontology]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train).to(device).type(torch.float)
X_test = torch.from_numpy(X_test).to(device).type(torch.float)
y_train = torch.from_numpy(y_train).to(device).type(torch.long)
y_test = torch.from_numpy(y_test).to(device).type(torch.long)

num_classes = torch.unique(y_train).shape[0]

space = {
    'C': hp.loguniform('C', np.log(0.001), np.log(100)),
    'lr': hp.loguniform('lr', np.log(1e-3), np.log(1e-1)),
    'dropout': hp.loguniform('dropout', np.log(1e-2), np.log(5e-1)),
    "hidden_dim": hp.quniform("hidden_dim", 2, 64, q=2)
}

def get_best_params(X_train, y_train):
    """
    Objective function for Hyperopt to find the best hyperparameters
    for Logistic Regression using stratified cross-validation.
    """

    def objective(params):
        clf = GNNModel(ontology_list, num_classes, params["lr"], params["C"], 500, params["dropout"], device=device)
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
        trials=trials
    )

    final_best_params = space_eval(space, best_params_raw)
    return final_best_params

best_params = get_best_params(X_train, y_train)
print(best_params)
# best_params = {
#     "C": 10,
#     "lr": 1e-2,
#     "dropout": 0.0,
#     "hidden_dim": 8
# }

acc_scores = []
for i in tqdm(range(10)):
    set_all_seeds(i)
    model = GNNModel(ontology_list, num_classes, best_params["lr"], best_params["C"], 500, best_params["dropout"], device=device, hidden_dim = best_params["hidden_dim"])
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    acc_scores.append(metrics.accuracy_score(y_test.cpu().numpy(), y_pred_test))
print(f"Accuracy: {np.mean(acc_scores)} +- {np.std(acc_scores)}")

class_labels = np.unique(y)
cm = metrics.confusion_matrix(y_test.cpu().numpy(), y_pred_test, labels=class_labels)
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
print(df_cm)


