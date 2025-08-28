import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import nn

from ibd_study.gnn_based.architecture.GNNOntologyLayerlist import GNN
from sklearn import metrics


class GNNModel:
    def __init__(self, ontology_list, num_classes, lr, weight_decay, max_epochs, dropout, device, hidden_dim = 1, num_layers = None, protein_embeddings= None, num_heads = 4):
        self.device = device
        self.model_params = {
            "ontology_list": ontology_list,
            "num_classes": num_classes,
            "dropout": dropout,
            # "protein_embeddings": protein_embeddings,
            "hidden_dim":  int(hidden_dim),
            "num_layers_clf": int(num_layers),
            "num_heads": int(num_heads),
            "act_function": nn.ReLU()
        }
        self.model = GNN(**self.model_params).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.best_model_state = None
        self.max_acc = float('-inf')
        self.patience_counter = 0
        self.patience = 100
        self.device = device

    def reinit_training(self):
        self.model = GNN(**self.model_params).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.best_model_state = None
        self.max_acc = float('-inf')
        self.patience_counter = 0

    def fit(self, features, y):
        # numpy arrays für Split
        features, y = features.cpu().numpy(), y.cpu().numpy()

        # einfacher stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            features, y, test_size=0.09, stratify=y, random_state=42
        )

        # Torch Tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).long()

        self.reinit_training()

        def closure():
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(X_train.to(self.device))
            data_loss = self.loss_fun(out, y_train.to(self.device))
            l2_penalty = (
                0.5 * self.weight_decay * self.model.out.weight.pow(2).sum()
                if "out" in self.model.state_dict()
                else 0
            )
            total_loss = data_loss + l2_penalty
            total_loss.backward()
            return total_loss

        best_epoch = 0
        best_acc = 0
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.optimizer.step(closure)

            # val accuracy
            acc = metrics.accuracy_score(
                y_val.cpu().numpy(),
                self.predict(X_val.to(self.device))
            )

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    # print(f"Early stopping at epoch {epoch}, best acc={best_acc:.4f}")
                    break

        # best model zurückladen
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_epoch, best_acc


    def fit_full(self, features, y):
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        features, y = features.cpu().numpy(), y.cpu().numpy()
        best_epochs = []
        for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(features, y), 1):
            self.reinit_training()
            X_train = features[train_index]
            y_train = y[train_index]
            X_val = features[test_index]
            y_val = y[test_index]

            X_train_tensor = torch.from_numpy(X_train).float()
            y_train_tensor = torch.from_numpy(y_train).long()
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).long()

            def closure():
                self.model.train()
                self.optimizer.zero_grad()
                out = self.model(X_train_tensor.to(self.device))
                data_loss = self.loss_fun(out, y_train_tensor.to(self.device))
                l2_penalty =  0.5 * self.weight_decay * self.model.out.weight.pow(2).sum() if "out" in self.model.state_dict() else 0
                total_loss = data_loss + l2_penalty
                total_loss.backward()
                return total_loss

            best_epoch_for_fold = 0
            for epoch in range(self.max_epochs):
                self.optimizer.step(closure)
                acc = metrics.accuracy_score(y_val_tensor, self.predict(X_val_tensor.to(self.device)))

                if acc > self.max_acc:
                    self.max_acc = acc
                    self.patience_counter = 0
                    best_epoch_for_fold = epoch
                    # self.best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        #print(f"Early stopping at epoch {epoch}. Best acc: {self.max_acc:.4f}")
                        best_epochs.append(best_epoch_for_fold)
                        break
        #print(best_epochs)
        self.reinit_training()
        X_train_tensor = torch.from_numpy(features).float()
        y_train_tensor = torch.from_numpy(y).long()
        def closure():
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(X_train_tensor.to(self.device))
            data_loss = self.loss_fun(out, y_train_tensor.to(self.device))
            l2_penalty = 0.5 * self.weight_decay * self.model.out.weight.pow(2).sum() if "out" in self.model.state_dict() else 0
            total_loss = data_loss + l2_penalty
            total_loss.backward()
            return total_loss

        for epoch in range(int(round(np.mean(np.array(best_epochs))))):
            self.optimizer.step(closure)

    def predict_proba(self, features):
        self.model.eval()  # CRITICAL: Set model to evaluation mode
        with torch.inference_mode():
            out = self.model(features.to(self.device))
            out = torch.nn.functional.softmax(out, dim=1)
            return out

    def predict(self, features):
        return torch.argmax(self.predict_proba(features), dim = 1).cpu().numpy()
