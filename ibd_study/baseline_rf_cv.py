import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

NESTED_CV = (10, 5)
# Load and prepare the data
data_df = pd.read_csv("data/transformed_df.csv")
sample_names = data_df.pop("index")
y = data_df.pop("condition")
X = data_df.values

print("Class distribution in the full dataset:")
print(np.unique(y, return_counts=True))

# Define the hyperparameter search space for Hyperopt
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 600, 50),
    'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample']),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_depth': hp.choice('max_depth_choice', [
        None,
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


def get_best_params(X_train_inner, y_train_inner):
    """
    Inner loop function for hyperparameter optimization using Hyperopt.
    This function is called within each fold of the outer cross-validation loop.
    """

    def objective(params):
        """Objective function for Hyperopt with Stratified K-Fold cross-validation."""
        # Ensure integer parameters are correctly typed
        for param_name in ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split']:
            if params.get(param_name) is not None:
                params[param_name] = int(params[param_name])

        clf = RandomForestClassifier(**params, random_state=42, n_jobs=3)
        # Inner CV for hyperparameter tuning
        skf = StratifiedKFold(n_splits=NESTED_CV[1], shuffle=True, random_state=42)

        acc = np.mean(cross_val_score(clf, X_train_inner, y_train_inner, cv=skf, scoring='accuracy', n_jobs=-1))

        # Hyperopt minimizes the loss, so we return negative accuracy
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best_params_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,  # Number of optimization iterations
        trials=trials,
        verbose=0  # Suppress verbose output from fmin
    )

    final_best_params = space_eval(space, best_params_raw)

    # Ensure final parameters are integers where required
    for param_name in ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split']:
        if final_best_params.get(param_name) is not None:
            final_best_params[param_name] = int(final_best_params[param_name])

    return final_best_params


# --- Nested Cross-Validation ---

# 1. Outer Loop for Model Evaluation
outer_cv = StratifiedKFold(n_splits=NESTED_CV[0], shuffle=True, random_state=42)

fold_accuracies = []
total_cm = None
class_labels = np.unique(y)

print("\nStarting Nested Cross-Validation...")
# Enumerate splits for tracking progress
for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n--- Processing Outer Fold {fold_idx}/{NESTED_CV[0]} ---")

    # Split data for the current outer fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 2. Inner Loop for Hyperparameter Tuning
    print("Finding best parameters using inner cross-validation...")
    best_params_for_fold = get_best_params(X_train, y_train)
    print(f"Best parameters for fold {fold_idx}: {best_params_for_fold}")

    # 3. Train model with best params on the outer training data
    model = RandomForestClassifier(**best_params_for_fold, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate on the outer test data
    y_pred_test = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred_test)
    fold_accuracies.append(accuracy)
    print(f"Accuracy on test set for fold {fold_idx}: {accuracy:.4f}")

    # Calculate and aggregate confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred_test, labels=class_labels)
    if total_cm is None:
        total_cm = cm
    else:
        total_cm += cm

# --- Final Evaluation Results ---
print("\n--- Nested Cross-Validation Complete ---")
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"\nAverage Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

print("\nAggregated Confusion Matrix across all folds:")
df_cm = pd.DataFrame(total_cm, index=class_labels, columns=class_labels)
print(df_cm)