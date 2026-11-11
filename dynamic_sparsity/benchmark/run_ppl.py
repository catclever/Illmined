import os
import sys
import math
import argparse
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamic_sparse_inference import DynamicAbsoluteSparseLinear

def compute_perplexity(model, tokens, seq_len=2048):
    total_loss = 0.0
    total_tokens = 0
    
    # Process in chunks of seq_len
    num_chunks = len(tokens) // seq_len
    if num_chunks == 0:
        num_chunks = 1
        seq_len = len(tokens)
        
    for i in range(num_chunks):
        chunk = tokens[i * seq_len : (i + 1) * seq_len]
        if len(chunk) < 2:
            continue
            
        x = mx.array([chunk[:-1]])
        y = mx.array([chunk[1:]])
        
        logits = model(x)
        # Logits shape: (1, seq_len-1, vocab_size)
        
        # Calculate cross entropy loss
        losses = nn.losses.cross_entropy(logits, y)
        loss_val = mx.sum(losses).item()
        
        total_loss += loss_val
        total_tokens += y.shape[1]
        
        if i % 10 == 0:
            print(f"  Processed {i}/{num_chunks} chunks...", end="\r")
            
    print(f"  Processed {num_chunks}/{num_chunks} chunks.")
    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)

def main():
    parser = argparse.ArgumentParser(description="Calculate Perplexity on WikiText-2")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8b", help="Model path")
    parser.add_argument("--max_tokens", type=int, default=100000, help="Max tokens to process")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    wiki_path = os.path.join(script_dir, "data", "wikitext_test.txt")
    
    if not os.path.exists(wiki_path):
        print(f"Dataset not found at {wiki_path}. Please run download_datasets.py first.")
        return

    print("Loading model...")
    model, tokenizer = load(args.model)
    model.eval()

    print("Tokenizing WikiText-2...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Tokenize and limit to max_tokens for speed
    tokens = tokenizer.encode(text)
    if len(tokens) > args.max_tokens:
        tokens = tokens[:args.max_tokens]
    print(f"Loaded {len(tokens)} tokens.")

    # 1. Run Dense
    print("\n--- Dense Model Perplexity ---")
    t0 = time.time()
    dense_ppl = compute_perplexity(model, tokens)
    print(f"Dense PPL: {dense_ppl:.4f} (took {time.time()-t0:.2f}s)")

    # 2. Setup Sparse
    print("\nPre-building sparse layers...")
    num_layers = len(model.model.layers)
    target_layers = list(range(10, 36))
    layer_sparsity = [0.0] * num_layers
    for idx, layer_idx in enumerate(target_layers):
        progress = idx / (len(target_layers) - 1)
        layer_sparsity[layer_idx] = 0.1 + progress * 0.6
        
    sparse_up_projs = {}
    orig_up_projs = {}
    
    for i, layer in enumerate(model.model.layers):
        orig_up_projs[i] = layer.mlp.up_proj
        if i in target_layers:
            orig_layer = layer.mlp.up_proj
            sparse_layer = DynamicAbsoluteSparseLinear(orig_layer, sparsity_ratio=layer_sparsity[i])
            sparse_up_projs[i] = sparse_layer

    def set_mode(is_sparse=False):
        for i, layer in enumerate(model.model.layers):
            if i in sparse_up_projs:
                if is_sparse:
                    layer.mlp.up_proj = sparse_up_projs[i]
                else:
                    layer.mlp.up_proj = orig_up_projs[i]

    # 3. Run Sparse
    print("\n--- Dynamic Absolute Sparse Model Perplexity ---")
    set_mode(is_sparse=True)
    t0 = time.time()
    sparse_ppl = compute_perplexity(model, tokens)
    print(f"Sparse PPL: {sparse_ppl:.4f} (took {time.time()-t0:.2f}s)")

    # Save Results
    report = f"# 困惑度 (Perplexity) 评测报告\n\n"
    report += f"- 数据集: WikiText-2 (Test Split, 取前 {len(tokens)} Tokens)\n"
    report += f"- 模型: {args.model}\n\n"
    report += "| 模型版本 | PPL (越低越好) | 相对劣化度 |\n"
    report += "|---|---|---|\n"
    degradation = (sparse_ppl - dense_ppl) / dense_ppl * 100
    report += f"| Dense 满血版 | **{dense_ppl:.4f}** | - |\n"
    report += f"| Sparse 动态稀疏 (10%-70%) | **{sparse_ppl:.4f}** | +{degradation:.2f}% |\n"

    report_path = os.path.join(script_dir, "ppl_report.md")
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"\nDone! Report saved to {report_path}")

if __name__ == "__main__":
    main()
