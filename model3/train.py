import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from model import InverseGainMLP
from data_split import train_loader, val_loader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scaling import y_tensor

# === 1. CİHAZ AYARI ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Kullanılan cihaz: {device}")

# === 2. MODEL, LOSS, OPTIMIZER ===
model = InverseGainMLP().to(device)

# Komponent bazlı ağırlıklandırma
# Önceki sürümde sabit değerler kullanılıyordu. Bunun yerine
# hedef değişkenlerin standart sapmalarına dayalı dinamik bir
# ağırlıklandırma tercih edildi. Böylece varyansı düşük
# komponentler (tahmini görece zor olanlar) eğitimde daha çok
# önem kazanacak.
with torch.no_grad():
    stds = torch.std(y_tensor, dim=0)
    component_weights = 1.0 / stds
    component_weights /= component_weights.sum()
    component_weights = component_weights.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

# === 3. EĞİTİM PARAMETRELERİ ===
epochs = 50
best_val_loss = float('inf')
patience_counter = 0
max_patience =40

train_losses = []
val_losses = []

# === 4. EĞİTİM DÖNGÜSÜ ===
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)

        loss_raw = F.smooth_l1_loss(y_pred, y_batch, reduction='none')  # [batch,10]
        loss = (loss_raw * component_weights).mean()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + len(train_loader) / len(train_loader))  # CosineAnnealWarmRestarts için

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # === VALİDASYON ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_val_pred = model(X_val)
            val_loss_raw = F.smooth_l1_loss(y_val_pred, y_val, reduction='none')
            loss_v = (val_loss_raw * component_weights).mean()
            val_loss += loss_v.item() * X_val.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # === EARLY STOPPİNG ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "model.pth")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print("✅ Early stopping triggered.")
            break

# === 5. LOSS GRAFİĞİ ===
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Weighted SmoothL1 Loss")
plt.title("Eğitim ve Doğrulama Kayıpları")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

print("✅ Eğitim tamamlandı. En iyi model 'model.pth' olarak kaydedildi.")
