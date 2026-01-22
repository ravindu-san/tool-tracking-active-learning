from torch.utils.data import TensorDataset, Subset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

class ToolTrackingDataset(Dataset):
    def __init__(self, Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y):
        self.Xt_acc = Xt_acc
        self.Xt_gyr = Xt_gyr
        self.Xt_mag = Xt_mag
        self.Xt_mic = Xt_mic
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = (
            self.Xt_acc[idx],
            self.Xt_gyr[idx],
            self.Xt_mag[idx],
            self.Xt_mic[idx],
        )
        return X, self.y[idx]



def stratified_sampling(Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y, test_size=0.15, val_size=0.15):
    # dataset = TensorDataset(X, y)
    dataset = ToolTrackingDataset(Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y)
    num_samples = len(dataset)
    all_idx = np.arange(num_samples)
    y_np = y.numpy()

    train_val_idx, test_idx, train_val_labels, test_labels = train_test_split(
        all_idx,
        y_np,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=y_np
    )

    val_size_ratio = val_size / (1 - test_size)

    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx,
        train_val_labels,
        test_size=val_size_ratio,
        random_state=42,
        shuffle=True,
        stratify=train_val_labels
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset


def stratified_sampling_train_val(Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y, val_size: float = 0.15,random_state: int = 42,):
    # dataset = TensorDataset(X, y)
    dataset = ToolTrackingDataset(Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y)
    y_np = y.detach().cpu().numpy()

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=val_size,
        stratify=y_np,
        shuffle=True,
        random_state=random_state,
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx)
