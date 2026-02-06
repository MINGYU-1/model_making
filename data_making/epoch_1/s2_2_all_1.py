import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Setup paths BEFORE importing custom modules
script_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 2. Now import your custom modules
from vae_earlystopping import EarlyStopping
from model.m2_bce import BCEcVAE
from model.m2_mse import MSEcVAE
from loss.l2_bce import l2_bce
from loss.l2_mse import l2_mse
from bce_metrics.bce_solve import eval_bce_metrics

results = {
    "random_state": [],
    "R2_BINARY": [],      # bce_binary * mse
    "R2_BCE_MSE": [],     # bce_prob * mse
    "R2_MSE": []          # mse only
}

# Run the loop
for i in np.random.randint(1, 100, size=20):
    # Use absolute paths using parent_dir
    x_data = np.load(os.path.join(parent_dir, 'data', 'metal.npy'))
    c_data = np.load(os.path.join(parent_dir, 'data', 'pre_re.npy'))
    
    x_train, x_test, c_train, c_test = train_test_split(x_data, c_data, random_state=i, test_size=0.4)
    x_val, x_test, c_val, c_test = train_test_split(x_test, c_test, random_state=i, test_size=0.5)
    
    x_scaler, c_scaler = MinMaxScaler(), MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)
    c_train = c_scaler.fit_transform(c_train)
    x_val, x_test = [x_scaler.transform(x) for x in [x_val, x_test]]
    c_val, c_test = [c_scaler.transform(c) for c in [c_val, c_test]]

    x_train, x_val, x_test = [torch.tensor(x, dtype=torch.float32) for x in [x_train, x_val, x_test]]
    c_train, c_val, c_test = [torch.tensor(c, dtype=torch.float32) for c in [c_train, c_val, c_test]]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(TensorDataset(x_train, c_train), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val, c_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, c_test), batch_size=64, shuffle=False)

    x_dim, c_dim = x_train.shape[1], c_train.shape[1]

    # --- BCE Model Training ---
    model_bce = BCEcVAE(x_dim, c_dim, z_dim=8).to(device)
    early_stopping = EarlyStopping(patience=40, min_delta=1e-9)
    optimizer = optim.Adam(model_bce.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(1, 601):
        model_bce.train()
        t_loss = 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            logit, mu, logvar = model_bce(x, c)
            loss_dict = l2_bce(logit, x, mu, logvar)
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss += loss_dict['loss'].item()
        
        model_bce.eval()
        v_loss = 0
        with torch.no_grad():
            for vx, vc in val_loader:
                vx, vc = vx.to(device), vc.to(device)
                v_logit, v_mu, v_logvar = model_bce(vx, vc)
                v_loss += l2_bce(v_logit, vx, v_mu, v_logvar)['loss'].item()
        
        if early_stopping(v_loss/len(val_loader), model_bce): break

    early_stopping.load_best_model(model_bce)
    model_bce.eval()
    
    # --- MSE Model Training ---
    model_mse = MSEcVAE(x_dim, c_dim, z_dim=8).to(device)
    early_stopping_mse = EarlyStopping(patience=40, min_delta=1e-9)
    optimizer_mse = optim.Adam(model_mse.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(1, 801):
        model_mse.train()
        t_loss = 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            optimizer_mse.zero_grad()
            x_hat, mu, logvar = model_mse(x, c)
            loss_dict = l2_mse(x_hat, x, mu, logvar)
            loss_dict['loss'].backward()
            optimizer_mse.step()
            t_loss += loss_dict['loss'].item()
            
        model_mse.eval()
        v_loss = 0
        with torch.no_grad():
            for vx, vc in val_loader:
                vx, vc = vx.to(device), vc.to(device)
                vh, vm, vl = model_mse(vx, vc)
                v_loss += l2_mse(vh, vx, vm, vl)['loss'].item()
        
        if early_stopping_mse(v_loss/len(val_loader), model_mse): break

    early_stopping_mse.load_best_model(model_mse)
    model_mse.eval()

    # --- Evaluation ---
    all_bce_logits, all_mse_logits, all_true = [], [], []
    with torch.no_grad():
        for xt, ct in test_loader:
            xt, ct = xt.to(device), ct.to(device)
            b_logit, _, _ = model_bce(xt, ct)
            m_logit, _, _ = model_mse(xt, ct)
            all_bce_logits.append(b_logit.cpu().numpy())
            all_mse_logits.append(m_logit.cpu().numpy())
            all_true.append(xt.cpu().numpy())

    bce_logits = np.vstack(all_bce_logits)
    mse_logits = np.vstack(all_mse_logits)
    x_true_scaled = np.vstack(all_true)

    bce_prob = 1 / (1 + np.exp(-bce_logits))
    bce_binary = (bce_prob >= 0.5).astype(np.float32)

    x_hat_fin = x_scaler.inverse_transform(mse_logits)
    x_true = x_scaler.inverse_transform(x_true_scaled)
    
    final_x_hat = x_hat_fin * bce_binary
    final_x_sig = x_hat_fin * bce_prob

    results["random_state"].append(int(i))
    results["R2_BINARY"].append(float(r2_score(x_true.flatten(), final_x_hat.flatten())))
    results["R2_BCE_MSE"].append(float(r2_score(x_true.flatten(), final_x_sig.flatten())))
    results["R2_MSE"].append(float(r2_score(x_true.flatten(), x_hat_fin.flatten())))

# Save once at the end
save_path = os.path.join(script_dir, "results_r2.json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Saved results to:", save_path)