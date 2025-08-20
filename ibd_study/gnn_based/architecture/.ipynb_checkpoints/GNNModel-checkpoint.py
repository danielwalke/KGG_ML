import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold

from ibd_study.gnn_based.architecture.GNN import GNN
from sklearn import metrics


class GNNModel:
    def __init__(self, ontology_list, num_classes, lr, C, max_epochs, dropout, device, hidden_dim = 1, num_layers = None, protein_embeddings= None):
        self.device = device
        self.model_params = {
            "ontology_list": ontology_list,
            "num_classes": num_classes,
            "dropout": dropout,
            "protein_embeddings": protein_embeddings,
            "hidden_dim":  int(hidden_dim),
            "num_layers": int(num_layers)
        }
        self.model = GNN(**self.model_params).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.C = C
        self.max_epochs = max_epochs
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.best_model_state = None
        self.max_acc = float('-inf')
        self.patience_counter = 0
        self.patience = 50
        self.device = device

    def reinit_training(self):
        self.model = GNN(**self.model_params).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_model_state = None
        self.max_acc = float('-inf')
        self.patience_counter = 0

    def fit(self, features, y):
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
                l2_penalty = 0.5 / self.C * self.model.out.weight.pow(2).sum()
                total_loss = data_loss + l2_penalty
                total_loss.backward()
                return total_loss

            for epoch in range(self.max_epochs):
                self.optimizer.step(closure)
                acc = metrics.accuracy_score(y_val_tensor, self.predict(X_val_tensor.to(self.device)))

                if acc > self.max_acc:
                    self.max_acc = acc
                    self.patience_counter = 0
                    # self.best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        #print(f"Early stopping at epoch {epoch}. Best acc: {self.max_acc:.4f}")
                        best_epochs.append(epoch)
                        break
        self.reinit_training()
            # if self.best_model_state:
            #     self.model.load_state_dict(self.best_model_state)
        X_train_tensor = torch.from_numpy(features).float()
        y_train_tensor = torch.from_numpy(y).long()
        def closure():
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(X_train_tensor.to(self.device))
            data_loss = self.loss_fun(out, y_train_tensor.to(self.device))
            l2_penalty = 0.5 / self.C * self.model.out.weight.pow(2).sum()
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
