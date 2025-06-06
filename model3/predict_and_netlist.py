import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from scipy.stats import skew, kurtosis
from model import InverseGainMLP

# ----------------------------------------
# 1. Ayarlar ve Dosya Yolları
# ----------------------------------------
CSV_PATH      = "filtered_data.csv"
SCALER_X_PATH = "scalers/scaler_X.pkl"
SCALER_Y_PATH = "scalers/scaler_y.pkl"
MODEL_PATH    = "model.pth"
OUTPUT_ASC    = "cs_amplifier_with_plot.asc"

# ----------------------------------------
# 2. Model ve Scaler’ları Yükle
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InverseGainMLP().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

# ----------------------------------------
# 3. CSV’den Rastgele Bir Satır Seç ve Gain Verisini Al
# ----------------------------------------
df = pd.read_csv(CSV_PATH)
gain_cols = [c for c in df.columns if c.startswith("gain_")]

random_idx = np.random.choice(df.index)
gain_row = df.loc[random_idx, gain_cols].values  # shape: (255,)

# ----------------------------------------
# 4. Seçilen Satırın Gain Profili Grafiğini Göster
# ----------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(gain_row, marker='o', linestyle='-', markersize=3)
plt.title(f"Satır {random_idx} için Gain Profili")
plt.xlabel("Gain İndeksi")
plt.ylabel("Gain Değeri")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------
# 5. Feature Engineering (Model’in Beklediği 273 Boyutlu Vektör)
# ----------------------------------------
gain_row_2d = gain_row.reshape(1, -1)  # (1, 255)

gain_max = gain_row_2d.max(axis=1, keepdims=True)
gain_min = gain_row_2d.min(axis=1, keepdims=True)
gain_mean = gain_row_2d.mean(axis=1, keepdims=True)
low_freq_mean  = gain_row_2d[:, :10].mean(axis=1, keepdims=True)
high_freq_mean = gain_row_2d[:, -10:].mean(axis=1, keepdims=True)
gain_drop      = (gain_row_2d[:, 0] - gain_row_2d[:, -1]).reshape(-1, 1)

gain_std  = gain_row_2d.std(axis=1, keepdims=True)
gain_skew = skew(gain_row_2d, axis=1).reshape(-1, 1)
gain_kurt = kurtosis(gain_row_2d, axis=1).reshape(-1, 1)

band1_mean = gain_row_2d[:, :5].mean(axis=1, keepdims=True)
band2_mean = gain_row_2d[:, 5:20].mean(axis=1, keepdims=True)
band3_mean = gain_row_2d[:, 20:50].mean(axis=1, keepdims=True)

# RMS ve türev tabanlı özellikler (scaling.py ile aynı mantık)
band1_rms = np.sqrt(np.mean(np.square(gain_row_2d[:, :5]), axis=1, keepdims=True))
band2_rms = np.sqrt(np.mean(np.square(gain_row_2d[:, 5:20]), axis=1, keepdims=True))
band3_rms = np.sqrt(np.mean(np.square(gain_row_2d[:, 20:50]), axis=1, keepdims=True))

first_deriv = np.diff(gain_row_2d, axis=1)
deriv_mean = first_deriv.mean(axis=1, keepdims=True)
deriv_std = first_deriv.std(axis=1, keepdims=True)
deriv_abs_mean = np.mean(np.abs(first_deriv), axis=1, keepdims=True)

X_extra = np.hstack([
    gain_max, gain_min, gain_mean,
    low_freq_mean, high_freq_mean,
    gain_drop,
    gain_std, gain_skew, gain_kurt,
    band1_mean, band2_mean, band3_mean,
    band1_rms, band2_rms, band3_rms,
    deriv_mean, deriv_std, deriv_abs_mean
])  # (1, 18)

X_full = np.hstack([gain_row_2d, X_extra])  # (1, 273)

# ----------------------------------------
# 6. Girdiyi Scaler’dan Geçir ve Torch Tensor’a Dönüştür
# ----------------------------------------
X_scaled = scaler_X.transform(X_full)                      # (1, 273)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# ----------------------------------------
# 7. Model ile Tahmin Yap ( ölçekli ) ve Inverse Transform
# ----------------------------------------
with torch.no_grad():
    y_scaled = model(X_tensor).cpu().numpy()                # (1, 10)

y_orig = scaler_y.inverse_transform(y_scaled)              # (1, 10)

component_cols = ['rin', 'rg1', 'rg2', 'rd', 'rs', 'cs', 'c1', 'c2', 'rl', 'vdd']
predicted = {component_cols[i]: float(y_orig[0, i]) for i in range(10)}

# Konsola tahminleri yazdır
print(f"\n🔧 Satır {random_idx} için Tahmin Edilen Komponent Değerleri:")
for comp, val in predicted.items():
    unit = "Ω" if comp in ["rin","rg1","rg2","rd","rs","rl"] else ("F" if comp in ["cs","c1","c2"] else "V")
    print(f"  • {comp}: {val:.6g} {unit}")

# ----------------------------------------
# 8. LTspice ASC Dosyasını Oluştur
# ----------------------------------------
Rin = predicted['rin']
RG1 = predicted['rg1']
RG2 = predicted['rg2']
RD  = predicted['rd']
RS  = predicted['rs']
CS  = predicted['cs']
C1  = predicted['c1']
C2  = predicted['c2']
RL  = predicted['rl']
VDD = predicted['vdd']
MOS_MODEL = "Si7336ADP"  # Örnek MOSFET; gerekirse değiştirebilirsiniz

netlist = f"""Version 4.1
SHEET 1 1116 680
WIRE 336 -240 256 -240
WIRE 256 -224 256 -240
WIRE 336 -224 336 -240
WIRE 336 -128 336 -144
WIRE 336 -128 256 -128
WIRE 400 -128 336 -128
WIRE 256 -112 256 -128
WIRE 400 -112 400 -128
WIRE 400 -16 400 -32
WIRE 480 -16 400 -16
WIRE 576 -16 544 -16
WIRE 400 0 400 -16
WIRE 576 16 576 -16
WIRE 48 80 16 80
WIRE 160 80 128 80
WIRE 256 80 256 -32
WIRE 256 80 224 80
WIRE 352 80 256 80
WIRE 400 112 400 96
WIRE 496 112 400 112
WIRE 576 112 576 96
WIRE 16 128 16 80
WIRE 256 128 256 80
WIRE 400 128 400 112
WIRE 496 128 496 112
WIRE 16 224 16 208
WIRE 256 224 256 208
WIRE 400 224 400 208
WIRE 496 224 496 192
FLAG 256 224 0
FLAG 16 224 0
FLAG 576 112 0
FLAG 400 224 0
FLAG 256 -224 0
FLAG 16 80 in
IOPIN 16 80 In
FLAG 576 -16 out
IOPIN 576 -16 Out
FLAG 496 224 0
SYMBOL voltage 16 112 R0
WINDOW 123 24 124 Left 2
WINDOW 39 0 0 Left 0
SYMATTR Value2 AC 1
SYMATTR InstName V1
SYMATTR Value 0
SYMBOL cap 224 64 R90
WINDOW 0 0 32 VBottom 2
WINDOW 3 32 32 VTop 2
SYMATTR InstName C1
SYMATTR Value {C1:.6g}
SYMBOL res 240 -128 R0
SYMATTR InstName RG1
SYMATTR Value {RG1:.6g}
SYMBOL res 240 112 R0
SYMATTR InstName RG2
SYMATTR Value {RG2:.6g}
SYMBOL nmos 352 0 R0
SYMATTR InstName M1
SYMATTR Value {MOS_MODEL}
SYMBOL res 384 -128 R0
SYMATTR InstName RD
SYMATTR Value {RD:.6g}
SYMBOL cap 544 -32 R90
WINDOW 0 0 32 VBottom 2
WINDOW 3 32 32 VTop 2
SYMATTR InstName C2
SYMATTR Value {C2:.6g}
SYMBOL res 560 0 R0
SYMATTR InstName RL
SYMATTR Value {RL:.6g}
SYMBOL voltage 336 -240 R0
SYMATTR InstName VDD
SYMATTR Value {VDD:.6g}
SYMBOL res 384 112 R0
SYMATTR InstName RS
SYMATTR Value {RS:.6g}
SYMBOL cap 480 128 R0
SYMATTR InstName CS
SYMATTR Value {CS:.6g}
SYMBOL res 144 64 R90
WINDOW 0 0 56 VBottom 2
WINDOW 3 32 56 VTop 2
SYMATTR InstName Rin
SYMATTR Value {Rin:.6g}
TEXT 448 -232 Left 2 !.ac dec 100 1 2G
"""

with open(OUTPUT_ASC, "w") as f:
    f.write(netlist)

print(f"\n✅ LTspice ASC dosyası oluşturuldu: {OUTPUT_ASC}")
print("\n------ Netlist Önizleme (ilk 15 satır) ------\n")
print("\n".join(netlist.splitlines()[:15]))
