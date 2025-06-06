import torch
import torch.nn as nn

class InverseGainMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 1. Ortak Paylaşımlı Katmanlar (Shared MLP) ---
        self.shared = nn.Sequential(
            nn.Linear(267, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True)
        )

        # --- 2. Bileşenler Arası Bilgi Birleşimi (Fusion) ---
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # --- 3. Çıkış Head'leri (Her bir komponent için) ---
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            ) for _ in range(10)
        ])

    def forward(self, x):
        # x: [batch_size, 267]
        shared_out = self.shared(x)   # [batch_size, 128]
        fused = self.fusion(shared_out)  # [batch_size, 64]
        outputs = [head(fused) for head in self.heads]  # Liste: 10 × [batch_size, 1]
        return torch.cat(outputs, dim=1)  # [batch_size, 10]

# Test amaçlı çalıştırıldığında model yapısını ve parametre sayısını yazdır
if __name__ == "__main__":
    model = InverseGainMLP()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔢 Toplam parametre sayısı: {total_params:,}")
