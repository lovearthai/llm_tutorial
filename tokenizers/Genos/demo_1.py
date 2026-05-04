import os
from transformers import AutoTokenizer

# 1. 明确模型路径 (请确保此目录下有你提到的含有单碱基 vocab 的 tokenizer.json)
model_path = "Genos-1.2B" 

print("="*60)
print("--- [步骤 1] 正在加载 Genos 单碱基 Tokenizer ---")
print("="*60)

# 加载分词器
# Genos 作为单碱基模型，其 tokenizer_config.json 里的类通常是 PreTrainedTokenizerFast
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("\n【💡 知识点：Genos 的单碱基特性】")
print("1. 词表构造：Genos 的词表（Vocab）非常精简，核心只有 A, C, G, T, N。")
print("2. 无合并逻辑：由于 merges 列表为空，它不会像 DNABERT-2 那样把 'AT' 合并。")
print("3. 对应关系：1 个碱基 = 1 个 Token = 1 个 ID。")

# 2. 准备输入
sequence = "GATCGATCGATC"
print(f"\n--- [步骤 2] 输入碱基序列 ---")
print(f"原始序列: {sequence}")
print(f"序列长度: {len(sequence)} bp")

# 3. 编码过程
# encode 会自动处理特殊 Token
input_ids = tokenizer.encode(sequence)

print(f"\n--- [步骤 3] 编码结果 (Encoding) ---")
print(f"Token IDs: {input_ids}")
print("说明：由于没有碱基合并，Token IDs 的数量（除去首尾特殊 Token）应严格等于碱基长度。")

# 4. 解码过程
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print(f"\n--- [步骤 4] 词法切分预览 (Tokenization) ---")
print(f"切分细节: {tokens}")

# 5. 可视化对齐
print(f"\n--- [步骤 5] 序列对齐还原 ---")
# 去掉特殊的 [CLS], [SEP] 等（具体视 Genos 的特殊 Token 定义而定）
# 假设词表开头是 [PAD], [UNK], [CLS], [SEP], [MASK]
core_tokens = tokens[1:-1] 

print(" -> ".join(core_tokens))
print("\n[分析]")
print(f"序列长度为 {len(sequence)}，切出的核心 Token 数量也是 {len(core_tokens)}。")
print("解释：Genos 严格遵循‘单碱基’逻辑，没有任何碱基被合并，这保证了模型对每个位置的碱基都有独立的注意力。")
