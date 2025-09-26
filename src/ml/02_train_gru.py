# Train a GRU to predict next-hour volume from the last 24 hours.
# Inputs:
#   artifacts/ml/windows.npz  -> X_train, y_train, X_val, y_val, X_test, y_test, t_*, s_*, window
#   artifacts/ml/scalers.csv  -> per-site mean/std for inverse-scaling
# Outputs:
#   artifacts/ml/gru.pt                  (trained model weights)
#   artifacts/predictions_gru.csv        (site_id, datetime, predicted/true volume)

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- paths ---
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
ART_ML = ART / "ml"

# --- small helpers ---
def load_windows():
    npz = np.load(ART_ML / "windows.npz", allow_pickle=True)
    X_train = npz["X_train"]; y_train = npz["y_train"]
    X_val   = npz["X_val"];   y_val   = npz["y_val"]
    X_test  = npz["X_test"];  y_test  = npz["y_test"]
    t_train = npz["t_train"]; t_val   = npz["t_val"]; t_test = npz["t_test"]
    s_train = npz["s_train"]; s_val   = npz["s_val"]; s_test = npz["s_test"]
    window  = int(npz["window"][0])
    scalers = pd.read_csv(ART_ML / "scalers.csv")  # site_id, mean, std
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            t_train, t_val, t_test, s_train, s_val, s_test, window, scalers)

class GRURegressor(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden,
                          num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x shape: (B, T, 1)
        out, _ = self.gru(x)     # out: (B, T, hidden)
        last = out[:, -1, :]     # take last time step (B, hidden)
        yhat = self.head(last)   # (B, 1)
        return yhat

def make_loader(X, y, batch=64, shuffle=False):
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)

def train(model, train_loader, val_loader, epochs=15, lr=1e-3, device="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * len(xb)
            n_train += len(xb)
        train_mse = train_loss_sum / max(1, n_train)

        # --- validate ---
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss_sum += loss.item() * len(xb)
                n_val += len(xb)
        val_mse = val_loss_sum / max(1, n_val)

        print(f"Epoch {ep:02d} | train MSE {train_mse:.4f} | val MSE {val_mse:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def invert_scale(vec_scaled, site_ids, scalers_df):
    means = scalers_df.set_index("site_id")["mean"].to_dict()
    stds  = scalers_df.set_index("site_id")["std"].to_dict()
    out = []
    for z, sid in zip(vec_scaled, site_ids):
        m = float(means.get(int(sid), 0.0))
        s = float(stds.get(int(sid), 1.0))
        out.append(float(z) * s + m)
    return np.array(out, dtype=float)

def main():
    # 0) load data
    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     t_tr, t_va, t_te, s_tr, s_va, s_te, window, scalers) = load_windows()

    # 1) device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 2) build model + loaders (beginner defaults)
    model = GRURegressor(input_size=1, hidden=64, num_layers=1).to(device)
    train_loader = make_loader(X_tr, y_tr, batch=64, shuffle=True)
    val_loader   = make_loader(X_va, y_va, batch=64, shuffle=False)
    test_loader  = make_loader(X_te, y_te, batch=64, shuffle=False)

    # 3) train
    model = train(model, train_loader, val_loader, epochs=15, lr=1e-3, device=device)

    # 4) test predictions (in scaled space)
    model.eval()
    preds_scaled_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yhat = model(xb)                     # (B,1)
            preds_scaled_list.append(yhat.cpu().numpy().reshape(-1))
    preds_scaled = np.concatenate(preds_scaled_list, axis=0).reshape(-1)

    # 5) invert scaling to original units (vph)
    y_test_scaled = y_te.reshape(-1)  # scaled truth
    y_pred = invert_scale(preds_scaled, s_te, scalers)
    y_true = invert_scale(y_test_scaled, s_te, scalers)

    # 6) simple metrics
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    mape = float(np.mean(np.abs((y_pred - y_true) / np.clip(y_true, 1.0, None))) * 100.0)
    print(f"Test MAE: {mae:.2f} vph | MAPE: {mape:.2f}%")

    # 7) save model
    ART_ML.mkdir(parents=True, exist_ok=True)
    model_path = ART_ML / "gru.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved model:", model_path)

    # 8) save predictions (site_id, datetime, predicted, true) for inspection / routing
    out = pd.DataFrame({
        "site_id": s_te.astype(int),
        "datetime": pd.to_datetime(t_te.astype("datetime64[s]")),
        "predicted_volume_vph": y_pred,
        "true_volume_vph": y_true
    }).sort_values(["site_id", "datetime"])
    out_path = ART / "predictions_gru.csv"
    out.to_csv(out_path, index=False)
    print("Saved predictions:", out_path)

if __name__ == "__main__":
    main()
