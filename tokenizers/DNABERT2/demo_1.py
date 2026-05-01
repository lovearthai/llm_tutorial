import os
from transformers import AutoTokenizer

# 1. 明确模型路径
# 如果是本地加载，只需提供包含 tokenizer.json 等文件的文件夹路径
model_path = "zhihan1996/DNABERT-2-117M"

print("--- [步骤 1] 正在加载 Tokenizer ---")
# 解释加载的文件
print(f"正在从路径 '{model_path}' 读取配置...")
print("提示：此过程主要调用了目录下的 'tokenizer.json' (词表数据) 和 'tokenizer_config.json' (参数配置)。")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 准备输入
sequence = "GATCGATCGATC"
print(f"\n--- [步骤 2] 输入碱基序列 ---")
print(f"原始序列: {sequence}")
print(f"序列长度: {len(sequence)} bp")

# 3. 编码过程
# encode 是将文字转为 ID 的过程
input_ids = tokenizer.encode(sequence)

print(f"\n--- [步骤 3] 编码结果 (Encoding) ---")
print(f"Token IDs: {input_ids}")
print("说明：这些数字是模型真正能理解的token_id。")
print(f"  - 1 代表 {tokenizer.decode([1])} (开始标记)")
print(f"  - 2 代表 {tokenizer.decode([2])} (结束标记)")

# 4. 解码过程
# 将 ID 转回人类可读的切分块
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print(f"\n--- [步骤 4] 词法切分预览 (Tokenization) ---")
print(f"切分细节: {tokens}")

# 5. 可视化对齐
print(f"\n--- [步骤 5] 序列对齐还原 ---")
# 去掉特殊的 [CLS] 和 [SEP] 方便观察
core_tokens = tokens[1:-1]
print(" -> ".join(core_tokens))
print(f"解释：DNABERT-2 发现 '{core_tokens[1]}' 是一个高频出现的组合，所以把它切成了一个整体。")
