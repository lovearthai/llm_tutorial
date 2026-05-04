import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model import SimpleVQVAE

class MemmapDataset(Dataset):
    def __init__(self, file_path, num_chunks=100, chunk_size=12000):
        # 以只读内存映射模式加载 npy，不占用实际内存
        self.data = np.load(file_path, mmap_mode='r')
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        # 提取第 idx 个 chunk 并转换为 float32 tensor
        # 增加通道维度以匹配 Conv1d 输入: (1, 12000)
        chunk = self.data[idx].astype(np.float32)
        return torch.from_numpy(chunk).unsqueeze(0)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 超参数
    batch_size = 10
    epochs = 50
    lr = 1e-3

    # 数据准备
    dataset = MemmapDataset("example.npy", num_chunks=100, chunk_size=12000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型与优化器
    model = SimpleVQVAE(codebook_size=512, dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    print(f"正在 {device} 上启动训练...")

    model.train()
    for epoch in range(epochs):
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            # 前向传播
            recon, vq_loss = model(batch)
            
            # 计算总损失
            recon_loss = mse_loss(recon, batch)
            total_loss = recon_loss + vq_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Recon Loss: {epoch_recon_loss/len(dataloader):.6f} | "
              f"VQ Loss: {epoch_vq_loss/len(dataloader):.6f}")

    # 保存权重
    torch.save(model.state_dict(), "vqvae_weights.pth")
    print("训练结束，权重已保存为 vqvae_weights.pth")

if __name__ == "__main__":
    train()
