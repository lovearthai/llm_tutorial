import os
from transformers import AutoTokenizer

# 1. 明确模型路径
# 确保该目录下有 vocab.txt, tokenizer_config.json 等文件
model_path = "nucleotide-transformer-v2-50m-multi-species"

print("="*70)
print("--- [步骤 1] 正在实例化 EsmTokenizer ---")
print("="*70)
print(f"正在从路径 '{model_path}' 读取配置...")

# 加载分词器
# trust_remote_code=True 是必须的，因为 EsmTokenizer 的逻辑可能包含在仓库的 py 文件中
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# --- 知识点穿插：历史与架构 ---
print("\n【💡 焦帅提示：为什么这个模型没有 tokenizer.json？】")
print("1. 家族背景：该模型源自 Meta 的 ESM (Evolutionary Scale Modeling) 架构，专门为生物序列设计。")
print("2. 架构设计：")
print("   - 标准模型 (如 GPT-4)：使用 'tokenizer.json' (Rust 实现)，追求复杂的 BPE 子词合并。")
print("   - 生物模型 (如 NT-v2)：使用 'vocab.txt' (Python 实现)。")
print("   - 核心逻辑：DNA 序列非常规整。3-mer 模型只需要穷举 4^3=64 种组合，简单的词表映射比复杂的 BPE 更高效。")

# --- 知识点穿插：词表内幕 ---
print(f"\n【🔍 词表内幕：vocab.txt 是如何被读取的？】")
# 注意：EsmTokenizer 没有 .vocab 属性，必须使用 get_vocab()
try:
    current_vocab = tokenizer.get_vocab()
    print(f"成功获取词表！词表总大小: {len(current_vocab)}")
    
    # 展示几个典型的 6-mer
    sample_kmers = ["AAAAAA", "TGCAGC", "GATTTT"]
    print("词表示例:")
    for k in sample_kmers:
        if k in current_vocab:
            print(f"  - 碱基块 '{k}' -> 对应的 Token ID 是: {current_vocab[k]}")
except Exception as e:
    print(f"获取词表失败，报错信息: {e}")

# 2. 准备输入
sequence = "GATCGATCGATC"
print(f"\n--- [步骤 2] 输入碱基序列 ---")
print(f"原始序列: {sequence} (长度: {len(sequence)} bp)")

# 3. 编码过程
# encode 会自动加上 [CLS] (ID: 1) 和 [SEP] (ID: 2)
input_ids = tokenizer.encode(sequence)

print(f"\n--- [步骤 3] 编码结果 (Encoding) ---")
print(f"Token IDs: {input_ids}")
print("解释：这是模型在推理时真正看到的数字向量。数字 1 和 2 是特殊标记。")

# 4. 解码过程
# 将 ID 转回 3-mer 字符串
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print(f"\n--- [步骤 4] 词法切分预览 (Tokenization) ---")
print(f"切分细节: {tokens}")

# --- 知识点穿插：K-mer 切分逻辑 ---
print("\n【🧪 生物信息学逻辑：固定窗口切分】")
print("仔细观察：")
# 这里的 [1:-1] 是为了去掉首尾的 [CLS] 和 [SEP]
core_tokens = tokens[1:-1]
print(f"  序列被切分成了: {' -> '.join(core_tokens)}")
print(f"逻辑：EsmTokenizer 像一把手术刀，每隔 3 个碱基切一下。")
print(f"优势：这种 3-mer 方式比单碱基(1-mer)的感受野扩大了 3 倍，且完美对应生物学中的密码子。")

# 5. 总结对比表
print(f"\n--- [步骤 5] 总结：两种 Tokenizer 体系对比 ---")
print("-" * 65)
print(f"{'特性':<15} | {'标准文本模型 (如 Llama)':<22} | {'生物 DNA 模型 (如 NT-v2)':<20}")
print("-" * 65)
print(f"{'核心配置文件':<15} | {'tokenizer.json (Fast)':<22} | {'vocab.txt (Slow)':<20}")
print(f"{'实现语言':<15} | {'Rust (高性能)':<22} | {'Python (高定制)':<20}")
print(f"{'切分策略':<15} | {'BPE / 动态合并':<22} | {'K-mer / 固定切分':<20}")
print(f"{'典型类名':<15} | {'PreTrainedTokenizerFast':<22} | {'EsmTokenizer':<20}")
print("-" * 65)

print("\n🚀 运行提示：你可以尝试输入长度不是 3 的倍数的序列（如 'GATCA'），")
print("看看 EsmTokenizer 是会丢弃末尾碱基，还是将其识别为特殊的 Unknown token。")
