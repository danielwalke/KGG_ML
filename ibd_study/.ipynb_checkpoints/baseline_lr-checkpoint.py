import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from hyperopt import space_eval, STATUS_OK, Trials, fmin, tpe, hp
import datetime
import os
from logger import logging

# Load and prepare data
data_df = pd.read_csv("data/transformed_df.csv")
sample_names = data_df.pop("index")
y = data_df.pop("condition")
X = data_df.values

print("Class distribution in the full dataset:")
print(np.unique(y, return_counts=True))

# Split data, ensuring stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
# LR is sensitive to feature scaling, so this is an important step.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler fitted on the training data

# Hyperparameter search space for Logistic Regression
space = {
    'C': hp.loguniform('C', np.log(0.001), np.log(100)),
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
    'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),  # Only used for 'elasticnet' penalty
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'solver': 'saga'  # SAGA solver supports all listed penalties
}


def get_best_params(X_train, y_train):
    """
    Objective function for Hyperopt to find the best hyperparameters
    for Logistic Regression using stratified cross-validation.
    """

    def objective(params):
        # The 'l1_ratio' parameter is only used when penalty is 'elasticnet'
        if params['penalty'] != 'elasticnet':
            params.pop('l1_ratio', None)

        clf = LogisticRegression(**params, random_state=42, max_iter=1000)  # Increased max_iter for saga solver
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Using 3 splits for CV

        # Use scaled data for cross-validation
        acc = np.mean(cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1))

        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best_params_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    final_best_params = space_eval(space, best_params_raw)

    # Clean up params for final model
    if final_best_params.get('penalty') != 'elasticnet':
        final_best_params.pop('l1_ratio', None)

    return final_best_params


# Find the best parameters using the scaled training data
print("Searching for best hyperparameters...")
final_best_params = get_best_params(X_train_scaled, y_train)
print("Best parameters found:")
print(final_best_params)

# Train the final model with the best parameters on the scaled training data
model = LogisticRegression(**final_best_params, random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

## Evaluation
# Predict on the scaled test data
y_pred_test = model.predict(X_test_scaled)

# Calculate and print metrics
accuracy_score = metrics.accuracy_score(y_test, y_pred_test)
print(f"\nTest Accuracy: {accuracy_score:.4f}")

class_labels = np.unique(y)
cm = metrics.confusion_matrix(y_test, y_pred_test, labels=class_labels)
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
print("\nConfusion Matrix:")
print(df_cm)
log_message = (
    f"Model Run Results - {current_date}\n"
    f"  File: {file_name}\n"
    f"  Accuracy: {accuracy_score:.2f}\n"
    f"  Confusion Matrix: {df_cm}\n"
    f"--------------------------------------------------"
)
logging.info(log_message)