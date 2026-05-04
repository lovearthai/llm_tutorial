import torch
import numpy as np
from model import SimpleVQVAE

def infer():
    # 1. 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "vqvae_weights.pth"
    data_path = "example.npy"
    chunk_size = 12000
    
    # 2. 加载模型
    model = SimpleVQVAE(codebook_size=512, dim=512).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载权重文件: {model_path}")
    except FileNotFoundError:
        print(f"错误: 找不到权重文件 {model_path}")
        return

    model.eval()

    # 3. 读取数据
    data = np.load(data_path, mmap_mode='r')
    input_chunk = data[0].astype(np.float32)
    # Shape: (Batch=1, Channel=1, Length=12000)
    input_tensor = torch.from_numpy(input_chunk).unsqueeze(0).unsqueeze(0).to(device)

    print("\n--- [原始数据预览 (前 10 个采样点)] ---")
    print(input_chunk[:10])

    # 4. 推理生成 Token
    # 注意这里必须是 no_grad()
    with torch.no_grad():
        # Step A: 提取特征
        z = model.encoder(input_tensor)
        # Step B: 转换维度供 VQ 使用 (B, C, L) -> (B, L, C)
        z_permuted = z.permute(0, 2, 1)
        # Step C: 获取离散索引
        _, indices, _ = model.vq(z_permuted)

    # 5. 结果展示
    tokens = indices[0].cpu().numpy()
    
    print("\n--- [Token 化结果] ---")
    print(f"Token 序列总长度: {len(tokens)}")
    print(f"前 20 个 Token: \n{tokens[:20]}")

    # 6. 验证重建
    with torch.no_grad():
        recon, _ = model(input_tensor)
        recon_chunk = recon[0, 0].cpu().numpy()
        print("\n--- [重建数据预览 (前 10 个采样点)] ---")
        print(recon_chunk[:10])

if __name__ == "__main__":
    infer()
