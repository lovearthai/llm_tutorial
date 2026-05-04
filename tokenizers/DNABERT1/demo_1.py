import os
from transformers import AutoTokenizer

# 1. 明确模型路径
model_path = "DNA_bert_3"

print("--- [步骤 1] 正在加载 Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# --- 💡 关键修正：手动切分函数 ---
def seq2kmer(seq, k):
    """将序列转为带空格的 overlapping k-mer"""
    return " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])

# 2. 准备输入
raw_sequence = "GATCGATCGATC"
# DNABERT 必须要这种格式: "GAT ATC TCG CGA GAT..."
kmer_sequence = seq2kmer(raw_sequence, 3) 

print(f"\n--- [步骤 2] 输入处理 ---")
print(f"原始序列: {raw_sequence}")
print(f"预处理后: {kmer_sequence}")

# 3. 编码过程
input_ids = tokenizer.encode(kmer_sequence)

print(f"\n--- [步骤 3] 编码结果 ---")
print(f"Token IDs: {input_ids}")

# 4. 解码过程
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(f"\n--- [步骤 4] 词法切分预览 ---")
print(f"切分细节: {tokens}")

# 5. 可视化对齐
print(f"\n--- [步骤 5] 序列对齐还原 ---")
core_tokens = tokens[1:-1] # 去掉 [CLS] 和 [SEP]
print(" -> ".join(core_tokens))

print("\n【📌 总结 DNABERT 的坑】")
print("1. 它没有内置滑动窗口逻辑，必须手动把序列切成空格分隔的 k-mer 再喂给它。")
print("2. 它的 vocab.txt 里面只有像 'GAT' 这样的 3-mer，不认识长链。")
print("3. 这就是典型的 Overlapping 设计：步长(Stride)为 1，产生大量的冗余 Token。")
