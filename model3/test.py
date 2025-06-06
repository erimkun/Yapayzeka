import torch
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
from model import InverseGainMLP
from data_split import test_loader
from scaling import scaler_y, component_cols
from sklearn.metrics import mean_absolute_error, r2_score

# === CİHAZ AYARI ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧪 Test cihazı: {device}")

# === MODELİ YÜKLE ===
model = InverseGainMLP().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# === TEST TAHMİNİ ===
y_true_list, y_pred_list = [], []

with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test = X_test.to(device)
        preds = model(X_test)
        y_true_list.append(y_test.numpy())
        y_pred_list.append(preds.cpu().numpy())

# === BİRLEŞTİR ===
y_true = np.vstack(y_true_list)
y_pred = np.vstack(y_pred_list)

# === INVERSE TRANSFORM (log1p ve MinMax geri dönüşüm) ===
y_true_orig = scaler_y.inverse_transform(y_true)
y_pred_orig = scaler_y.inverse_transform(y_pred)

# === MAE ve R² HESAPLAMA ===
results = []
for i, col in enumerate(component_cols):
    mae = mean_absolute_error(y_true_orig[:, i], y_pred_orig[:, i])
    r2 = r2_score(y_true_orig[:, i], y_pred_orig[:, i])
    results.append({"Component": col, "MAE": mae, "R² Score": r2})

df_results = pd.DataFrame(results)
print("\n=== Component Bazlı MAE ve R² Skorları ===")
print(df_results.to_string(index=False))

# === OPSİYONEL: CSV olarak kaydet ===
df_results.to_csv("test_metrics.csv", index=False)
print("\n✅ test_metrics.csv dosyasına kaydedildi.")
