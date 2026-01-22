# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# %%
def random_time_mask_imu(x, mask_ratio=0.4):
    """
    x: (T, D)
    returns: mask (T,1)
    """
    T = x.shape[0]
    mask = (torch.rand(T, device=x.device) > mask_ratio).float()
    return mask.unsqueeze(-1)


def random_time_mask_audio(x, mask_ratio=0.4):
    """
    x: (T,)
    returns: mask (T,)
    """
    T = x.shape[0]
    return (torch.rand(T, device=x.device) > mask_ratio).float()


# %%
class TCNEncoder(nn.Module):
    def __init__(self, in_ch, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (T, D)
        x = x.transpose(0, 1).unsqueeze(0)  # (1, D, T)
        h = self.net(x)
        return h.squeeze(0).transpose(0, 1)  # (T, hidden)


# %%
class RawAudioEncoder(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (T,)
        """
        assert x.dim() == 1, f"Expected (T,), got {x.shape}"
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,T)
        h = self.net(x)
        return h.squeeze(0).transpose(0, 1)  # (T, hidden)


# %%
class TemporalPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=0)


# %%
class MultimodalMAE(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.pool = TemporalPool()

        # Encoders
        self.acc_enc = TCNEncoder(3)
        self.gyr_enc = TCNEncoder(3)
        self.mag_enc = TCNEncoder(3)
        self.mic_enc = RawAudioEncoder()

        # Fusion head
        self.fusion_proj = nn.Sequential(
            nn.Linear(64*4, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, acc, gyr, mag, mic):
        h_acc = self.acc_enc(acc)
        h_gyr = self.gyr_enc(gyr)
        h_mag = self.mag_enc(mag)
        h_mic = self.mic_enc(mic)

        z = self.fusion_proj(torch.cat([
            self.pool(h_acc),
            self.pool(h_gyr),
            self.pool(h_mag),
            self.pool(h_mic)
        ], dim=-1))

        return F.normalize(z, dim=-1), (h_acc, h_gyr, h_mag, h_mic)


# %%
class MaskedReconstruction(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.imu_head = nn.Linear(hidden, 3)
        self.audio_head = nn.Linear(hidden, 1)

    def forward(self, h_acc, h_gyr, h_mag, h_mic):
        return (
            self.imu_head(h_acc),
            self.imu_head(h_gyr),
            self.imu_head(h_mag),
            self.audio_head(h_mic),
        )


if __name__ == "__main__":
    # %%
    import sys
    import os
    import numpy as np
    import torch

    PROJECT_ROOT = os.path.abspath("../..")
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from al_data_process.data_preprocess import create_windowed_ts, map_class_label_to_idx
    from al_data_process.dataset import stratified_sampling_train_val

    window_length = 0.4
    overlap = 0.25

    invalid_classes = [14]
    Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y_org_label = create_windowed_ts(data_path="../../tool-tracking-data/",
                                                                        tool="electric_screwdriver",
                                                                        invalid_classes=invalid_classes,
                                                                        window_length = window_length,
                                                                        overlap = overlap)

    all_classes, counts = np.unique(y_org_label, return_counts=True)
    # print(len(y_org_label))
    # print(all_classes, counts)

    Xt_acc = Xt_acc[:, :, 1:].astype(np.float32)
    Xt_gyr = Xt_gyr[:, :, 1:].astype(np.float32)
    Xt_mag = Xt_mag[:, :, 1:].astype(np.float32)
    Xt_mic = Xt_mic[:, :, 1:].astype(np.float32)

    y = map_class_label_to_idx(y_org_label)

    train_set, test_set = stratified_sampling_train_val(torch.from_numpy(Xt_acc),
                                                    torch.from_numpy(Xt_gyr),
                                                    torch.from_numpy(Xt_mag),
                                                    torch.from_numpy(Xt_mic),
                                                    torch.tensor(y))


    X_pool = np.empty(len(train_set), dtype=object)
    y_pool = np.empty(len(train_set))
    for i in range(len(train_set)):
        X_pool[i] = train_set[i][0]
        y_pool[i] = train_set[i][1]


    X_test = np.empty(len(test_set), dtype=object)
    y_test = np.empty(len(test_set))
    for i in range(len(test_set)):
        X_test[i] = test_set[i][0]
        y_test[i] = test_set[i][1]


    # %%
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model = MultimodalMAE().to(device)
    recon = MaskedReconstruction().to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(recon.parameters()), lr=1e-3)

    num_epochs = 100
    model.train()
    recon.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for acc, gyr, mag, mic in X_pool:
            # Move to device
            acc = acc.to(device)
            gyr = gyr.to(device)
            mag = mag.to(device)
            mic = mic.to(device).squeeze()  # ensure 1D

            # Masking
            mask_acc = random_time_mask_imu(acc)
            mask_gyr = random_time_mask_imu(gyr)
            mask_mag = random_time_mask_imu(mag)
            mask_mic = random_time_mask_audio(mic)

            acc_m = acc * mask_acc
            gyr_m = gyr * mask_gyr
            mag_m = mag * mask_mag
            mic_m = mic * mask_mic

            # Forward
            _, (h_acc, h_gyr, h_mag, h_mic) = model(acc_m, gyr_m, mag_m, mic_m)

            # Reconstruction
            rec_acc, rec_gyr, rec_mag, rec_mic = recon(h_acc, h_gyr, h_mag, h_mic)

            # Loss (masked MSE)
            loss = (
                ((rec_acc - acc)**2 * (1 - mask_acc)).mean()
                + ((rec_gyr - gyr)**2 * (1 - mask_gyr)).mean()
                + ((rec_mag - mag)**2 * (1 - mask_mag)).mean()
                + ((rec_mic.squeeze(-1) - mic)**2 * (1 - mask_mic)).mean()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f}")



    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'recon_state_dict': recon.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }

    # Save to disk
    torch.save(checkpoint, "multimodal_mae.pth")
    print("Model saved successfully!")



    # %%
    model = MultimodalMAE().to(device)
    recon = MaskedReconstruction().to(device)

    checkpoint = torch.load("multimodal_mae.pth", map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    recon.load_state_dict(checkpoint['recon_state_dict'])

    model.eval()
    recon.eval()

    print("Models loaded and ready for inference.")

    # %%
    @torch.no_grad()
    def extract_embedding(acc, gyr, mag, mic):
        model.eval()
        acc, gyr, mag, mic = (
            acc.to(device),
            gyr.to(device),
            mag.to(device),
            mic.to(device).squeeze()
        )
        z, _ = model(acc, gyr, mag, mic)
        return z.cpu()

    # %%
    embeddings = []
    for acc, gyr, mag, mic in X_pool:
        z = extract_embedding(acc, gyr, mag, mic)
        embeddings.append(z)

    embeddings = np.array(embeddings)

    print(embeddings.shape)

    print(embeddings[0].shape)

    np.save("./data/X_pool_embeddings.npy", embeddings)


    # embeddings = []
    # for acc, gyr, mag, mic in X_test:
    #     z = extract_embedding(acc, gyr, mag, mic)
    #     embeddings.append(z)

    # embeddings = np.array(embeddings)

    # print(embeddings.shape)

    # print(embeddings[0].shape)

    # np.save("./data/X_pool_embeddings.npy", embeddings)

    # %%
    embeddings = np.load("./data/X_pool_embeddings.npy")
    print(embeddings.shape)

