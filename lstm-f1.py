import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from datatools import MeasurementDataReader, Tool, Config, MeasurementSeries, Measurement, DataTypes, Action
# from datatools import ACC, GYR, MAG, MIC, POS, VEL
# from datatools import to_ts_data
# from fhgutils import Segment, contextual_recarray_dtype, filter_ts_data
import numpy as np
# from seglearn.base import TS_Data
# from seglearn.pipe import Pype
# from fhgutils import filter_labels, one_label_per_window, summarize_labels
# from sklearn.model_selection import train_test_split
import wandb

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score

from al_data_process.data_preprocess import create_windowed_ts, map_class_label_to_idx
from al_data_process.dataset import stratified_sampling, stratified_sampling_train_val
from model.lstm_classif import LSTM_CLASSIF

torch.manual_seed(1337)

class FocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: torch.Tensor,
        gamma: float = 2.0,
        normalize: bool = True,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        class_counts = class_counts.float()
        total = class_counts.sum()

        alpha = total / (class_counts + eps)

        if normalize:
            alpha = alpha / alpha.mean()

        self.alpha = alpha  # [num_classes]

    def forward(self, logits, targets):
        device = logits.device
        alpha = self.alpha.to(device)

        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        alpha_t = alpha[targets]
        focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

def save_checkpoint(path, model, optimizer, epoch, best_val_f1, early_stopping, scheduler=None):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "early_stopping": {
            "best_score": early_stopping.best_score,
            "counter": early_stopping.counter,
        }
    }

    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    early_stopping = EarlyStopping()
    es_state = checkpoint.get("early_stopping", {})
    early_stopping.best_score = es_state.get("best_score", -float("inf"))
    early_stopping.counter = es_state.get("counter", 0)

    epoch = checkpoint.get("epoch", 0)
    best_val_f1 = checkpoint.get("best_val_f1", 0.0)

    print(f"Loaded checkpoint from {path}, resumed at epoch {epoch + 1}")
    return epoch, best_val_f1, early_stopping

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    num_epochs=50,
    patience=7,
    ckpt_dir="checkpoints",
    resume_path=None,
    save_every=5,
):

    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    if resume_path:
        start_epoch, best_val_f1, early_stopping = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
    else:
        start_epoch = 0
        best_val_f1 = 0.0
        early_stopping = EarlyStopping(patience=patience)

    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        # ================= TRAIN =================
        model.train()
        train_loss_total = 0.0

        # for (Xt_acc, Xt_gyr, Xt_mag, Xt_mic), y in train_loader:
        #     Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y = (
        #         Xt_acc.to(device),
        #         Xt_gyr.to(device),
        #         Xt_mag.to(device),
        #         Xt_mic.to(device),
        #         y.to(device),
        #     )
        #
        #     optimizer.zero_grad()
        #     logits = model(Xt_acc, Xt_gyr, Xt_mag, Xt_mic)
        for (Xt_acc, Xt_gyr, Xt_mag, Xt_mic), y in train_loader:
            X, y = (Xt_acc.to(device),Xt_gyr.to(device),Xt_mag.to(device),Xt_mic.to(device)),y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * y.size(0)

        train_loss = train_loss_total / len(train_loader.dataset)

        # ================= VALIDATION =================
        model.eval()
        val_loss_total = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            # for Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y in val_loader:
            #     Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y = (
            #         Xt_acc.to(device),
            #         Xt_gyr.to(device),
            #         Xt_mag.to(device),
            #         Xt_mic.to(device),
            #         y.to(device),
            #     )
            #
            #     logits = model(Xt_acc, Xt_gyr, Xt_mag, Xt_mic)
            for (Xt_acc, Xt_gyr, Xt_mag, Xt_mic), y in val_loader:
                X, y = (Xt_acc.to(device),Xt_gyr.to(device),Xt_mag.to(device),Xt_mic.to(device)),y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                val_loss_total += loss.item() * y.size(0)

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss = val_loss_total / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_acc = (np.array(val_preds) == np.array(val_labels)).mean()

        if scheduler:
            scheduler.step(val_f1)  # IMPORTANT: mode="max"

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"Val F1 (macro): {val_f1:.4f}"
        )

        # ================= SAVE BEST MODEL =================
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

            wandb.run.summary["best_val_f1"] = best_val_f1
            wandb.run.summary["best_val_acc"] = val_acc

            print(f"Best model updated! Macro F1: {best_val_f1:.4f}")

        # ================= CHECKPOINT =================
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                path=os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_f1=best_val_f1,
                early_stopping=early_stopping,
                scheduler=scheduler,
            )

        # ================= EARLY STOPPING =================
        early_stopping.step(val_f1)
        if early_stopping.should_stop:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best validation macro F1: {best_val_f1:.4f}")


def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # for (Xt_acc, Xt_gyr, Xt_mag, Xt_mic), y in test_loader:
        #     # Xt_acc, Xt_gyr, y = Xt_acc.to(device), Xt_gyr.to(device), y.to(device)
        #     Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y = Xt_acc.to(device), Xt_gyr.to(device), Xt_mag.to(device), Xt_mic.to(device), y.to(device)
        #     logits = model(Xt_acc, Xt_gyr, Xt_mag, Xt_mic)
        for (Xt_acc, Xt_gyr, Xt_mag, Xt_mic), y in test_loader:
            # X, y = X.to(device), y.to(device)
            X, y = (Xt_acc.to(device), Xt_gyr.to(device), Xt_mag.to(device), Xt_mic.to(device)), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f}")
    return avg_loss, accuracy


def test(model_path: str = "checkpoints/best_model.pth", test_loader=None, criterion=None, hidden_size=32, num_lstm_layers=1, dropout_p=0.5, device="cpu"):
    model = LSTM_CLASSIF(input_size=3, hidden_size=hidden_size, num_layers=num_lstm_layers,  num_classes=7,dropout_p=dropout_p).to(device)
    model.load_state_dict(torch.load(model_path))

    test_loss, test_acc = test_model(model, test_loader, criterion, device)

    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    # wandb.finish()


def evaluate_model(model, dataloader, device, best_model_path, log_to_wandb=False):
    """
    Loads a trained model and computes accuracy, precision, recall,
    F1-score, and confusion matrix on the given dataloader.
    """

    # Load model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # Run inference
    with torch.no_grad():
        # for Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y in dataloader:
        #     Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y = (
        #         Xt_acc.to(device),
        #         Xt_gyr.to(device),
        #         Xt_mag.to(device),
        #         Xt_mic.to(device),
        #         y.to(device)
        #     )
        #
        #     logits = model(Xt_acc, Xt_gyr, Xt_mag, Xt_mic)
        for (Xt_acc, Xt_gyr, Xt_mag, Xt_mic), y in dataloader:
            # X, y = (X.to(device), y.to(device))
            X, y = (Xt_acc.to(device), Xt_gyr.to(device), Xt_mag.to(device), Xt_mic.to(device)), y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = (all_preds == all_labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Optional W&B logging
    if log_to_wandb:
        wandb.run.summary["eval_accuracy"] = accuracy
        wandb.run.summary["eval_precision"] = precision
        wandb.run.summary["eval_recall"] = recall
        wandb.run.summary["eval_f1"] = f1
        wandb.run.summary["eval_confusion_matrix"] = cm.tolist()

        # W&B Confusion Matrix visualization
        wandb.log({
            "eval_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_preds
            )
        })

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    # BATCH_SIZE = 16
    # LR = 2e-3
    # num_lstm_layers = 1
    # hidden_size = 32
    # dropout_p = 0.3
    BATCH_SIZE = 16
    LR = 2e-3
    num_lstm_layers = 1
    hidden_size = 32
    dropout_p = 0.3
    is_test_set = True
    window_length = 0.4
    overlap = 0.25
    ckpt_dir = "checkpoints"

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    wandb_config = {
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "device": device,
        "num_lstm_layers": num_lstm_layers,
        "run": "lstm",
    }
    wandb.init(project="ADLTS", mode="disabled", config=wandb_config)

    invalid_classes = [14]
    Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y_org_label = create_windowed_ts(data_path="./tool-tracking-data/",
                                                                     tool="electric_screwdriver",
                                                                     invalid_classes=invalid_classes,
                                                                     window_length = window_length,
                                                                     overlap = overlap)

    all_classes, counts = np.unique(y_org_label, return_counts=True)
    print(all_classes, counts)

    Xt_acc = Xt_acc[:, :, 1:].astype(np.float32)
    Xt_gyr = Xt_gyr[:, :, 1:].astype(np.float32)
    Xt_mag = Xt_mag[:, :, 1:].astype(np.float32)
    Xt_mic = Xt_mic[:, :, 1:].astype(np.float32)

    y = map_class_label_to_idx(y_org_label)

    if is_test_set:
        train_dataset, val_dataset, test_dataset = stratified_sampling(torch.from_numpy(Xt_acc),
                                                                   torch.from_numpy(Xt_gyr),
                                                                   torch.from_numpy(Xt_mag),
                                                                   torch.from_numpy(Xt_mic),
                                                                   torch.tensor(y))
    else:
        train_dataset, val_dataset = stratified_sampling_train_val(torch.from_numpy(Xt_acc),
                                                                       torch.from_numpy(Xt_gyr),
                                                                       torch.from_numpy(Xt_mag),
                                                                       torch.from_numpy(Xt_mic),
                                                                       torch.tensor(y))


    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    if is_test_set :
        y_train = train_dataset.dataset[train_dataset.indices][-1].numpy()
    else:
        y_train = [sample[-1] for sample in train_dataset]
    classes, per_class_counts = np.unique(y_train, return_counts=True)

    # criterion = FocalLoss(torch.from_numpy(per_class_counts), gamma=3.0)
    def compute_class_weights(class_counts, beta=0.999):
        class_counts = torch.tensor(class_counts, dtype=torch.float)
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        return weights / weights.mean()

    print(classes, per_class_counts)
    weights = compute_class_weights(per_class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    model = LSTM_CLASSIF(input_size=3, hidden_size=hidden_size, num_layers=num_lstm_layers, num_classes=7, dropout_p=dropout_p)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=200,
        patience=7,
        ckpt_dir=ckpt_dir,
        resume_path=None,
        save_every=5
    )

    if is_test_set:
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        best_model_path = "checkpoints/best_model.pth"

        test(best_model_path, test_loader, criterion, hidden_size=hidden_size, num_lstm_layers=num_lstm_layers, dropout_p=dropout_p, device=device)

        model = LSTM_CLASSIF(input_size=3, hidden_size=hidden_size, num_layers=num_lstm_layers, num_classes=7,
                             dropout_p=dropout_p)

        metrics = evaluate_model(
            model=model,
            # dataloader=val_loader,
            dataloader=test_loader,
            device=device,
            best_model_path=os.path.join(ckpt_dir, "best_model.pth"),
            log_to_wandb=True
        )

    wandb.finish()
