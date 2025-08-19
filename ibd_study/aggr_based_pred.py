import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import space_eval, STATUS_OK, Trials, fmin, tpe, hp
from logger import logging
import datetime
import os

data_df = pd.read_csv("data/transformed_df.csv")
sample_names = data_df.pop("index")
y = data_df.pop("condition")
print(np.unique(y, return_counts=True))

proteins_over_samples_df = data_df.transpose().reset_index()
proteins_over_samples_df["index"] = proteins_over_samples_df["index"].astype(np.int64)
tax_edges = pd.read_csv("data/tax_edges.csv")

merged_proteins_over_sampled_df = pd.merge(tax_edges, proteins_over_samples_df, on="index")

order_level_features = merged_proteins_over_sampled_df.groupby("order").apply(np.mean, include_groups=False, axis = 0).transpose().iloc[7:]
phylum_level_features = merged_proteins_over_sampled_df.groupby("phylum").apply(np.mean, include_groups=False, axis = 0).transpose().iloc[7:]
genus_level_features = merged_proteins_over_sampled_df.groupby("genus").apply(np.mean, include_groups=False, axis = 0).transpose().iloc[7:]
species_level_features = merged_proteins_over_sampled_df.groupby("species").apply(np.mean, include_groups=False, axis = 0).transpose().iloc[7:]
X = np.hstack((phylum_level_features, genus_level_features, species_level_features))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 600, 50),
    'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample']),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_depth': hp.choice('max_depth_choice', [
        None,  # Default value
        hp.quniform('max_depth_int', 3, 30, 3)
    ]),
    'max_features': hp.choice('max_features_choice', [
        'sqrt',
        'log2',
        hp.uniform('max_features_float', 0.2, 1.0)
    ]),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
}


def get_best_params(X_train, y_train):
    def objective(params):
        """
        Objective function for Hyperopt.
        Takes a dictionary of hyperparameters and returns the value to minimize (negative AUROC).
        Includes Stratified K-Fold cross-validation.
        """
        for param_name in ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split']:
            if params[param_name] is not None:
                params[param_name] = int(params[param_name])
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=3)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        acc = np.mean(cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1))

        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best_params_raw = fmin(
        fn=objective,  # The function to optimize
        space=space,  # The search space
        algo=tpe.suggest,  # The Tree-structured Parzen Estimator algorithm
        max_evals=100,  # The number of iterations
        trials=trials  # The object to store trial results
    )
    final_best_params = space_eval(space, best_params_raw)
    for param_name in ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split']:
        if final_best_params[param_name] is not None:
            final_best_params[param_name] = int(final_best_params[param_name])
    return final_best_params

final_best_params = get_best_params(X_train, y_train)
model = RandomForestClassifier(**final_best_params, random_state=42)
model.fit(X_train, y_train)


## Evaluation
y_pred_test = model.predict(X_test)
accuracy_score = metrics.accuracy_score(y_test, y_pred_test)
print(accuracy_score)

class_labels = np.unique(y)
cm = metrics.confusion_matrix(y_test, y_pred_test, labels=class_labels)
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
print(df_cm)
log_message = (
    f"Model Run Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"  File: {os.path.basename(__file__)}\n"
    f"  Accuracy: {accuracy_score:.2f}\n"
    f"  Confusion Matrix: {df_cm}\n"
    f"--------------------------------------------------"
)
logging.info(log_message)

