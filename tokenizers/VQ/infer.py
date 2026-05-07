import torch
import numpy as np
from model import SimpleVQVAE
import matplotlib.pyplot as plt

def save_reconstruction_plot(original, reconstructed, save_path="reconstruction_comparison.png", plot_len=1000):
    """
    绘制并保存原始信号与重建信号的对比图
    """
    # 确保绘制长度不超过数据实际长度
    actual_len = min(plot_len, len(original))
    
    plt.figure(figsize=(15, 6))
    
    # 绘制原始信号
    plt.plot(original[:actual_len], label='Original Signal', alpha=0.7, color='blue')
    
    # 绘制重建信号
    plt.plot(reconstructed[:actual_len], label='Reconstructed Signal', alpha=0.7, color='red', linestyle='--')
    
    plt.title(f"Signal Reconstruction Comparison (First {actual_len} samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # 关闭画板释放内存
    print(f"\n[成功] 对比图已保存为: {save_path}")

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
    # 7. 调用绘图函数
    save_reconstruction_plot(input_chunk, recon_chunk)
if __name__ == "__main__":
    infer()
