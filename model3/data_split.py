import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from scaling import X_tensor, y_tensor

# === 1. Dataset oluştur ===
full_dataset = TensorDataset(X_tensor, y_tensor)
total_samples = len(full_dataset)

# === 2. Bölme oranları ===
train_len = int(0.70 * total_samples)
val_len   = int(0.15 * total_samples)
test_len  = total_samples - train_len - val_len  # kalan örnekler test'e gider

# === 3. Dataset'leri böl ===
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42)  # reproducibility için sabit seed
)

# === 4. Dataloader'lar oluştur ===
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === 5. Kontrol çıktısı ===
print("✅ Veri bölme tamamlandı:")
print(f"🔹 Toplam örnek sayısı: {total_samples}")
print(f"🔸 Eğitim seti: {len(train_dataset)}")
print(f"🔸 Doğrulama seti: {len(val_dataset)}")
print(f"🔸 Test seti: {len(test_dataset)}")
print(f"📦 Batch size: {batch_size}")
print(f"📤 Eğitim batch sayısı: {len(train_loader)}")

# === 6. Export edilecek nesneler ===
__all__ = ['train_loader', 'val_loader', 'test_loader']
