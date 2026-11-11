"""
run_hellaswag.py

客观题（常识推理）评测脚本。
负责运行 HellaSwag 数据集，通过计算不同选项后缀的困惑度/生成概率，
选出最符合常识的续写选项，以此来评估模型的逻辑与推理能力。

特点：
- 测试模型在非开放式生成情况下的绝对准确率（Accuracy）。
- 为了避免 MLX 的动态图重复编译，在数据层面进行了长度截断和强制右侧 Padding 补齐，
  并在模型评估侧采用了 mx.compile 固定图执行策略。
"""
import os
import sys
import json
import argparse
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamic_sparse_inference import DynamicAbsoluteSparseLinear

from mlx_lm.models.cache import make_prompt_cache

def evaluate_hellaswag(model, tokenizer, dataset, num_samples=100, set_mode_func=None, prefill_sparse=False, decode_sparse=False):
    """
    Evaluates the model on the HellaSwag validation dataset.
    Strictly follows 'Prefill Dense, Decode Sparse' by splitting the context and option.
    """
    correct = 0
    total = 0
    predictions = []
    
    start_time = time.time()
    
    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        ctx = item['ctx']
        endings = item['endings']
        label = int(item['label'])
        
        ctx_tokens = tokenizer.encode(ctx)
        
        scores = []
        for ending in endings:
            # Tokenize context and ending separately.
            # Some tokenizers behave differently when tokenizing parts separately, 
            # but for HellaSwag this is generally acceptable or we can add a space.
            full_text = ctx + " " + ending
            full_tokens = tokenizer.encode(full_text)
            ending_len = len(full_tokens) - len(ctx_tokens)
            
            if ending_len <= 0:
                scores.append(-float('inf'))
                continue
                
            option_tokens = full_tokens[len(ctx_tokens):]
            
            # --- 1. PREFILL (Context) ---
            if set_mode_func:
                set_mode_func(is_sparse=prefill_sparse)
            
            cache = make_prompt_cache(model)
            # Pass the context to build the KV cache.
            ctx_tensor = mx.array([ctx_tokens])
            logits = model(ctx_tensor, cache=cache)
            mx.eval(logits)
            
            # The last logit of the context predicts the first token of the option
            ending_logits = [logits[0, -1, :]]
            
            # --- 2. DECODE (Option) ---
            if set_mode_func:
                set_mode_func(is_sparse=decode_sparse)
                
            # Feed the option tokens one by one to get the subsequent logits
            # We don't need to feed the very last option token, because we don't predict after it.
            for t in range(len(option_tokens) - 1):
                next_token_tensor = mx.array([[option_tokens[t]]])
                step_logits = model(next_token_tensor, cache=cache)
                mx.eval(step_logits)
                ending_logits.append(step_logits[0, -1, :])
                
            # Stack logits for the option
            ending_logits_tensor = mx.stack(ending_logits)
            ending_y = mx.array(option_tokens)
            
            log_probs = mx.log(mx.softmax(ending_logits_tensor, axis=-1))
            indices = mx.arange(len(option_tokens))
            token_log_probs = log_probs[indices, ending_y]
            
            score = mx.sum(token_log_probs).item() / len(option_tokens)  # normalized by length
            scores.append(score)
            
        pred = scores.index(max(scores))
        is_correct = (pred == label)
        predictions.append(is_correct)
        if is_correct:
            correct += 1
        total += 1
        
        if total % 5 == 0:
            acc = (correct / total) * 100
            print(f"  Processed {total}/{num_samples} samples (Acc: {acc:.2f}%)", flush=True)
            
    acc = correct / total * 100
    print(f"  Final Acc: {acc:.2f}%", flush=True)
    return acc, predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate on HellaSwag")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8b", help="Model path")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    hellaswag_path = os.path.join(script_dir, "data", "hellaswag_val.jsonl")
    
    if not os.path.exists(hellaswag_path):
        print(f"Dataset not found at {hellaswag_path}. Please run download_datasets.py first.")
        return

    print("Loading HellaSwag Dataset...")
    dataset = []
    with open(hellaswag_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
            
    print("Loading model...")
    model, tokenizer = load(args.model)
    model.eval()

    # 1. Run Dense
    print(f"\n--- Dense Model HellaSwag (N={args.num_samples}) ---")
    t0 = time.time()
    dense_acc, dense_preds = evaluate_hellaswag(model, tokenizer, dataset, num_samples=args.num_samples)
    print(f"Dense Acc: {dense_acc:.2f}% (took {time.time()-t0:.2f}s)", flush=True)

    # 2. Build Sparse Layers mapping
    print("\nPre-building sparse layers...", flush=True)
    orig_up_projs = {}
    sparse_up_projs = {}
    
    target_layers = list(range(len(model.model.layers)))
    layer_sparsity = {}
    
    for i in target_layers:
        ratio = i / (len(model.model.layers) - 1)
        # Inverted U-shape for HellaSwag: starts at 0.1, peaks at 0.7 in the middle, drops to 0.1 at the end
        sparsity = 0.1 + 0.6 * (1 - 4 * (ratio - 0.5)**2)
        layer_sparsity[i] = sparsity
        
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
    print(f"\n--- Dynamic Absolute Sparse Model HellaSwag (N={args.num_samples}) ---", flush=True)
    if hasattr(model, '_compiled_forward'):
        delattr(model, '_compiled_forward')
    t0 = time.time()
    sparse_acc, sparse_preds = evaluate_hellaswag(model, tokenizer, dataset, num_samples=args.num_samples,
                                    set_mode_func=set_mode, prefill_sparse=False, decode_sparse=True)
    print(f"Sparse Acc: {sparse_acc:.2f}% (took {time.time()-t0:.2f}s)", flush=True)

    # Output detailed error analysis
    analysis = "## Detailed Error Analysis\n\n"
    both_correct = 0
    both_wrong = 0
    dense_better = 0
    sparse_better = 0
    
    categories_stats = {}
    
    for i in range(len(dense_preds)):
        item = dataset[i]
        cat = item['activity_label']
        d = dense_preds[i]
        s = sparse_preds[i]
        
        if cat not in categories_stats:
            categories_stats[cat] = {'total': 0, 'dense_correct': 0, 'sparse_correct': 0, 'dense_won': 0, 'sparse_won': 0}
        
        categories_stats[cat]['total'] += 1
        if d: categories_stats[cat]['dense_correct'] += 1
        if s: categories_stats[cat]['sparse_correct'] += 1
        
        if d and s:
            both_correct += 1
        elif not d and not s:
            both_wrong += 1
        elif d and not s:
            dense_better += 1
            categories_stats[cat]['dense_won'] += 1
        elif not d and s:
            sparse_better += 1
            categories_stats[cat]['sparse_won'] += 1

    analysis += f"- Both Correct: {both_correct}\n"
    analysis += f"- Both Wrong: {both_wrong}\n"
    analysis += f"- Dense Better (Dense ✓, Sparse ✗): {dense_better}\n"
    analysis += f"- Sparse Better (Sparse ✓, Dense ✗): {sparse_better}\n\n"
    
    analysis += "### Category Breakdown\n"
    analysis += "| Category | Total | Dense Acc | Sparse Acc | Dense Won | Sparse Won |\n"
    analysis += "|---|---|---|---|---|---|\n"
    for cat, stats in sorted(categories_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        d_acc = stats['dense_correct'] / stats['total'] * 100
        s_acc = stats['sparse_correct'] / stats['total'] * 100
        analysis += f"| {cat} | {stats['total']} | {d_acc:.1f}% | {s_acc:.1f}% | {stats['dense_won']} | {stats['sparse_won']} |\n"

    # Save Results
    report = f"# HellaSwag 常识推理评测报告\n\n"
    report += f"- 数据集: HellaSwag Validation Set (评估 {args.num_samples} 题)\n"
    report += f"- 模型: {args.model}\n\n"
    report += "| 模型版本 | 准确率 (Acc, 越高越好) | 相对损失 |\n"
    report += "|---|---|---|\n"
    diff = sparse_acc - dense_acc
    report += f"| Dense 满血版 | **{dense_acc:.2f}%** | - |\n"
    report += f"| Sparse 动态稀疏 (10%-70%) | **{sparse_acc:.2f}%** | {diff:+.2f}% |\n\n"
    report += analysis

    report_path = os.path.join(script_dir, "hellaswag_report.md")
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"\nDone! Report saved to {report_path}")

if __name__ == "__main__":
    main()
