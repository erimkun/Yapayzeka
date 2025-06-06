import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from scipy.stats import skew, kurtosis

# === 1. CSV YÜKLE ===
df = pd.read_csv("filtered_data.csv")

# === 2. GİRİŞ: gain sütunları ===
gain_cols = [col for col in df.columns if col.startswith("gain_")]
X_raw = df[gain_cols].values  # (n_samples, 255)

# === 3. ÖZNİTELİK MÜHENDİSLİĞİ (Feature Engineering) ===
# Mevcut öznitelikler:
gain_max = X_raw.max(axis=1, keepdims=True)
gain_min = X_raw.min(axis=1, keepdims=True)
gain_mean = X_raw.mean(axis=1, keepdims=True)
low_freq_mean = X_raw[:, :10].mean(axis=1, keepdims=True)
high_freq_mean = X_raw[:, -10:].mean(axis=1, keepdims=True)
gain_drop = (X_raw[:, 0] - X_raw[:, -1]).reshape(-1, 1)

# Yeni eklenen öznitelikler:
#  - gain_std: Tüm kazanç eğrisinin standart sapması
#  - gain_skew / gain_kurt: Çarpıklık ve basıklık değerleri
gain_std = X_raw.std(axis=1, keepdims=True)
gain_skew = skew(X_raw, axis=1).reshape(-1, 1)
gain_kurt = kurtosis(X_raw, axis=1).reshape(-1, 1)

# --- Ek Öznitelikler: Bant RMS ve türev istatistikleri ---
# Bant RMS hesaplamaları (farklı frekans aralıklarının enerji seviyesi)
band1_rms = np.sqrt(np.mean(np.square(X_raw[:, :5]), axis=1, keepdims=True))
band2_rms = np.sqrt(np.mean(np.square(X_raw[:, 5:20]), axis=1, keepdims=True))
band3_rms = np.sqrt(np.mean(np.square(X_raw[:, 20:50]), axis=1, keepdims=True))

# Birinci dereceden türev (yaklaşık eğri eğimi)
first_deriv = np.diff(X_raw, axis=1)
deriv_mean = first_deriv.mean(axis=1, keepdims=True)
deriv_std = first_deriv.std(axis=1, keepdims=True)
deriv_abs_mean = np.mean(np.abs(first_deriv), axis=1, keepdims=True)

# Band ortalamaları (örnek olarak 3 bant):
band1_mean = X_raw[:, :5].mean(axis=1, keepdims=True)    # ilk 5 frekans bölgesi
band2_mean = X_raw[:, 5:20].mean(axis=1, keepdims=True)  # 5–20 bölgesi
band3_mean = X_raw[:, 20:50].mean(axis=1, keepdims=True) # 20–50 bölgesi

# Tüm öznitelikleri yatay olarak birleştir:
X_extra = np.hstack([
    gain_max, gain_min, gain_mean,
    low_freq_mean, high_freq_mean,
    gain_drop,
    gain_std, gain_skew, gain_kurt,
    band1_mean, band2_mean, band3_mean,
    # RMS değerleri ve türev istatistikleri
    band1_rms, band2_rms, band3_rms,
    deriv_mean, deriv_std, deriv_abs_mean
])  # → (n_samples, 18 ek özellik)

X_full = np.hstack([X_raw, X_extra])  # (n_samples, 255 + 18 = 273)

# === 4. GİRİŞ SCALING ===
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_full)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# === 5. ÇIKIŞ: devre elemanları (ilk 10 sütun) ===
component_cols = ['rin', 'rg1', 'rg2', 'rd', 'rs', 'cs', 'c1', 'c2', 'rl', 'vdd']
y_df = df[component_cols].copy()

# === 6. Outlier Clipping (vdd için 1. ve 99. persentil) ===
vdd_lower = y_df['vdd'].quantile(0.01)
vdd_upper = y_df['vdd'].quantile(0.99)
y_df['vdd'] = y_df['vdd'].clip(vdd_lower, vdd_upper)
print(f"✅ 'vdd' clipped to [{vdd_lower:.3f}, {vdd_upper:.3f}] range.")

# === 7. Log-transform (sadece yüksek varyanslı komponentler) ===
cols_to_log = ['rin', 'rg1', 'rg2', 'rd', 'rs', 'rl']
for col in cols_to_log:
    y_df[col] = np.log1p(y_df[col])

# === 8. ÇIKIŞ SCALING ===
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_df.values)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# === 9. SCALER DOSYALARINI KAYDET ===
os.makedirs("scalers", exist_ok=True)
joblib.dump(scaler_X, "scalers/scaler_X.pkl")
joblib.dump(scaler_y, "scalers/scaler_y.pkl")
print("✅ Scaler dosyaları 'scalers/' klasörüne kaydedildi.")

# === 10. KONTROL ÇIKTISI ===
print("✅ Scaling tamamlandı:")
print("🔸 Giriş boyutu (X):", X_tensor.shape)    # örn: (258489, 273)
print("🔸 Çıkış boyutu (y):", y_tensor.shape)    # örn: (258489, 10)
print("🔧 Giriş sütunu sayısı:", X_tensor.shape[1])
print("🔧 Çıkış komponentleri:", component_cols)

# === 11. EXPORT ===
__all__ = ['X_tensor', 'y_tensor', 'scaler_X', 'scaler_y', 'component_cols']
