import random

def generate_bpe_corpus(filename, num_lines=1000, seq_len=100):
    # 定义一些“高频模式”（模拟生物学基序，如 TATA-box 或特定重复序列）
    motifs = ["TATAAA", "GGCCGG", "ATATATAT"]
    bases = ['A', 'C', 'G', 'T']
    
    with open(filename, 'w') as f:
        for _ in range(num_lines):
            # 随机生成基础序列
            seq = ''.join(random.choices(bases, k=seq_len))
            
            # 随机在序列中插入一些高频模式，方便 BPE 学习合并
            if random.random() > 0.5:
                insert_pos = random.randint(0, seq_len - 1)
                motif = random.choice(motifs)
                seq = seq[:insert_pos] + motif + seq[insert_pos:]
            
            f.write(seq + "\n")

generate_bpe_corpus("genome_corpus.txt")
print("文件 genome_corpus.txt 已生成！")
