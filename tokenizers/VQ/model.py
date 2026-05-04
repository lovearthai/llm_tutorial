import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

class SimpleVQVAE(nn.Module):
    def __init__(self, codebook_size=512, dim=512):
        super().__init__()
        
        # Encoder: 1 -> 128 -> 256 -> 512
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, dim, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(dim),
        )

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = codebook_size,
            decay = 0.8,
            commitment_weight = 1.
        )

        # Decoder: 512 -> 256 -> 128 -> 1
        self.decoder = nn.Sequential(
            # --- 关键修改：添加 output_padding=4 ---
            # 这样 2400 长度才能完美还原回 12000
            nn.ConvTranspose1d(
                dim, 256, 
                kernel_size=25, 
                stride=5, 
                padding=12, 
                output_padding=4, # 补齐 12000 - 11996 = 4
                bias=False
            ),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 1, kernel_size=5, padding=2, bias=True)
        )

    def forward(self, x):
        z = self.encoder(x)
        
        z = z.permute(0, 2, 1)
        quantized, indices, vq_loss = self.vq(z)
        quantized = quantized.permute(0, 2, 1)
        
        x_recon = self.decoder(quantized)
        
        return x_recon, vq_loss
