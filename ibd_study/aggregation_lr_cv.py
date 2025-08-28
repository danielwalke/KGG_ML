import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from hyperopt import space_eval, STATUS_OK, Trials, fmin, tpe, hp
import datetime
import os
from logger import logging

NESTED_CV = (10, 5)
data_df = pd.read_csv("data/transformed_df.csv")
sample_names = data_df.pop("index")
y = data_df.pop("condition")
proteins_over_samples_df = data_df.transpose().reset_index()
proteins_over_samples_df["index"] = proteins_over_samples_df["index"].astype(np.int64)

ko_edges = pd.read_csv("data/function_edges.csv")
go_edges = pd.read_csv("data/function_edges_go.csv")
ko_edges["index"] = ko_edges["index"].astype(np.int64)
go_edges["index"] = go_edges["index"].astype(np.int64)

merged_proteins_over_sampled_df_ko = pd.merge(ko_edges, proteins_over_samples_df, on="index")
merged_proteins_over_sampled_df_go = pd.merge(go_edges, proteins_over_samples_df, on="index")
ko_level_features = merged_proteins_over_sampled_df_ko.groupby("trg").apply(np.sum, include_groups=False, axis = 0).transpose().iloc[1:].values
go_level_features = merged_proteins_over_sampled_df_go.groupby("trg").apply(np.sum, include_groups=False, axis = 0).transpose().iloc[1:].values
print(ko_level_features.shape)
print(go_level_features.shape)
X = np.hstack((ko_level_features, go_level_features))

print("Class distribution in the full dataset:")
print(np.unique(y, return_counts=True))

space = {
    'C': hp.loguniform('C', np.log(0.001), np.log(100)),
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
    'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'solver': 'saga'
}


def get_best_params(X_train_inner, y_train_inner):
    def objective(params):
        if params['penalty'] != 'elasticnet':
            params.pop('l1_ratio', None)

        clf = LogisticRegression(**params, random_state=42, max_iter=1000)
        skf = KFold(n_splits=NESTED_CV[1], shuffle=True, random_state=42)

        acc = np.mean(cross_val_score(clf, X_train_inner, y_train_inner, cv=skf, scoring='accuracy', n_jobs=-1))

        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best_params_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=np.random.default_rng(42),
        verbose=0
    )

    final_best_params = space_eval(space, best_params_raw)

    if final_best_params.get('penalty') != 'elasticnet':
        final_best_params.pop('l1_ratio', None)

    return final_best_params


outer_cv = KFold(n_splits=NESTED_CV[0], shuffle=True, random_state=42)

fold_accuracies = []
total_cm = None
class_labels = np.unique(y)

print("\nStarting Nested Cross-Validation for Logistic Regression...")

for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X), 1):
    print(f"\n--- Processing Outer Fold {fold_idx}/{NESTED_CV[0]} ---")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Finding best parameters using inner cross-validation...")
    best_params_for_fold = get_best_params(X_train_scaled, y_train)
    print(f"Best parameters for fold {fold_idx}: {best_params_for_fold}")

    model = LogisticRegression(**best_params_for_fold, random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)
    accuracy = metrics.accuracy_score(y_test, y_pred_test)
    fold_accuracies.append(accuracy)
    print(f"Accuracy on test set for fold {fold_idx}: {accuracy:.4f}")

    cm = metrics.confusion_matrix(y_test, y_pred_test, labels=class_labels)
    if total_cm is None:
        total_cm = cm
    else:
        total_cm += cm

    log_message = (
        f"Model Run Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  File: {os.path.basename(__file__)} (Fold: {fold_idx} / {NESTED_CV[0]})\n"
        f"  Accuracy: {accuracy:.4f}\n"
        f"--------------------------------------------------"
    )
    logging.info(log_message)

print("\n--- Nested Cross-Validation Complete ---")
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"\nAverage Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

print("\nAggregated Confusion Matrix across all folds:")
df_cm = pd.DataFrame(total_cm, index=class_labels, columns=class_labels)
print(df_cm)
log_message = (
    f"Model Run Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"  File: {os.path.basename(__file__)}\n"
    f"  Accuracy: {mean_accuracy:.4f}± {std_accuracy:.4f}\n"
    f"  Confusion Matrix: {df_cm}\n"
    f"--------------------------------------------------"
)
logging.info(log_message)