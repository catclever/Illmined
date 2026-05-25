'''
File: build_calibration_dataset.py
功能: 构建用于提取协方差矩阵的校准数据集 (Calibration Corpus)。
说明: 从大规模 parquet 数据中过滤出长度在 500-1500 字符的高质量连续文本，随机采样 3000 条，用以充分展开语言模型的时序特征流。
'''

import pandas as pd
import json
import random
import os

def build_dataset():
    input_file = "data/Basic_ZH/chunked_mixed_omni.parquet"
    output_file = "data/calibration_corpus_3k_omni.jsonl"
    
    print(f"正在加载 {input_file} ...")
    df = pd.read_parquet(input_file, engine='pyarrow')
    print(f"总行数: {len(df)}")
    
    valid_texts = []
    
    # 遍历数据，拼接 chunks
    for idx, row in df.iterrows():
        chunks = row['chunks']
        # 过滤掉 <EOS> 标记，或替换为换行
        text = " ".join([str(c).replace('<EOS>', '\n').strip() for c in chunks if str(c).strip()])
        
        # 字符长度控制在 500 ~ 1500 之间（大致相当于 400 ~ 1200 Tokens）
        if 500 <= len(text) <= 1500:
            valid_texts.append(text)
            
    print(f"符合长度条件 (500~1500字) 的文本数量: {len(valid_texts)}")
    
    # 随机采样 3000 条
    sample_size = 3000
    if len(valid_texts) > sample_size:
        sampled_texts = random.sample(valid_texts, sample_size)
    else:
        sampled_texts = valid_texts
        
    print(f"采样后文本数量: {len(sampled_texts)}")
    
    # 保存为 jsonl
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in sampled_texts:
            record = {"text": text}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"成功导出至 {output_file}，大小: {os.path.getsize(output_file)/1024/1024:.2f} MB")

if __name__ == "__main__":
    build_dataset()
