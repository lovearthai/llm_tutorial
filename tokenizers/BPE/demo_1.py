from tokenizers import Tokenizer, models, trainers
import os

# 1. 初始化 BPE 模型
# models.BPE() 是基础架构，它告诉程序我们要用“字节对编码”逻辑
tokenizer = Tokenizer(models.BPE())

print("--- [步骤 1] 配置 BPE 训练器 ---")
# vocab_size: 最终词表的大小。BPE 会一直合并直到达到这个数字。
# initial_alphabet: 初始字母表，确保 A, C, G, T 永远在词表里。
trainer = trainers.BpeTrainer(
    vocab_size=1000, 
    initial_alphabet=["A", "C", "G", "T"],
    show_progress=True  # 显示你刚才看到的进度条
)

# 2. 准备文件
files = ["genome_corpus.txt"]
if not os.path.exists(files[0]):
    print(f"错误：找不到 {files[0]}，请先运行生成脚本！")
    exit()

print(f"\n--- [步骤 2] 开始训练 (从 {files[0]} 学习模式) ---")
# 这一步会执行：统计频率 -> 寻找最高频 Pair -> 合并 -> 重复
tokenizer.train(files, trainer)

# 3. 提取词表 (Vocab)
# 词表是一个字典，Key 是 Token 字符串，Value 是 ID
vocab = tokenizer.get_vocab()
# 我们按 ID 进行排序，这样可以看到合并的先后顺序
# ID 0-3 通常是 A, C, G, T，之后的 ID 都是合并出来的
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

print("\n--- [步骤 3] 词表进化预览 (前 20 个 Token) ---")
print(f"{'Token ID':<10} | {'Token 内容':<15} | {'说明'}")
print("-" * 50)
for token, token_id in sorted_vocab[:20]:
    category = "初始碱基" if token_id < 4 else "BPE 合并产生"
    print(f"{token_id:<10} | {token:<15} | {category}")

# 4. 编码测试
# 让我们看看 BPE 是如何“阅读”一段新序列的
test_seq = "TATAAAGCCGATATATAT"
print(f"\n--- [步骤 4] 编码测试 ---")
print(f"测试序列: {test_seq}")

output = tokenizer.encode(test_seq)
print(f"编码后的 Token IDs: {output.ids}")
print(f"切分后的 Tokens:    {output.tokens}")

# 5. 逻辑解释
print("\n--- [步骤 5] 结果分析 ---")
first_merge = sorted_vocab[4][0] if len(sorted_vocab) > 4 else "N/A"
print(f"1. 算法发现最频繁相邻的两个碱基是: '{first_merge}'，因此给它分配了 ID 4。")
print(f"2. 如果你的 Tokens 中出现了长字符串（如 'TATAAA'），说明该模式在语料中极其常见。")
print(f"3. 相比于固定的 3-mer，BPE 这种切分是‘变长’的，更符合生物基序的特征。")
