"""
run_benchmark.py

生成式（开放式）问答评测脚本 (1D 列稀疏版)
负责调用 MT-Bench 测试集，并通过 Ruby Judge Service（LLM 裁判）对比 
Dense（满血版）与 Dynamic Column Sparse（动态列稀疏版）大模型在开放式生成任务中的表现。
"""
import os
import sys
import time
import json
import argparse
import pickle
import urllib.request
import urllib.error
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamic_1d_sparse_inference import DynamicColumnSparseLinear, generate_custom

JUDGE_URL = "http://localhost:4567/evaluate"

def load_questions(jsonl_path, max_questions=10):
    questions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append({
                'id': data['question_id'],
                'category': data['category'],
                'prompt': data['turns'][0]
            })
            if len(questions) >= max_questions:
                break
    return questions

def judge_evaluate(question, ans_a, ans_b):
    payload = {
        "mode": "pairwise",
        "question": question,
        "answer_a": ans_a,
        "answer_b": ans_b
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(JUDGE_URL, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.URLError as e:
        print(f"  [Error] Judge service failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark Dense vs Dynamic Column Sparse")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8b", help="Model path")
    parser.add_argument("--num_questions", type=int, default=5, help="Number of questions to test")
    args = parser.parse_args()

    # Load Questions from original directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    q_path = os.path.join(script_dir, "../../dynamic_sparsity/benchmark/question.jsonl")
    if not os.path.exists(q_path):
        print(f"Dataset not found at {q_path}.")
        return
        
    questions = load_questions(q_path, args.num_questions)
    
    # Load Model
    print("Loading model...")
    model, tokenizer = load(args.model)
    model.eval()

    num_layers = len(model.model.layers)
    target_layers = list(range(num_layers))
    
    layer_sparsity = [0.0] * num_layers
    for layer_idx in target_layers:
        ratio = layer_idx / (num_layers - 1)
        layer_sparsity[layer_idx] = 0.1 + 0.6 * (1 - 4 * (ratio - 0.5)**2)
    
    sparse_up_projs = {}
    orig_up_projs = {}
    
    print("Pre-building sparse layers...")
    perm_path = os.path.join(script_dir, "../outputs/qwen_1d_col_cluster_perm.pkl")
    with open(perm_path, 'rb') as f:
        permutation_maps = pickle.load(f)

    for i, layer in enumerate(model.model.layers):
        orig_up_projs[i] = layer.mlp.up_proj
        if i in target_layers and i < len(permutation_maps):
            orig_layer = layer.mlp.up_proj
            
            sparsity = layer_sparsity[i]
            perm_idx = mx.array(permutation_maps[i])
            shuffled_weight = mx.array(orig_layer.weight[:, perm_idx])
            
            temp_linear = nn.Linear(shuffled_weight.shape[1], shuffled_weight.shape[0], bias=False)
            temp_linear.weight = shuffled_weight
            
            sparse_layer = DynamicColumnSparseLinear(temp_linear, sparsity_ratio=sparsity)
            
            from dynamic_1d_sparse_inference import PermutedSparseLinear
            sparse_up_projs[i] = PermutedSparseLinear(sparse_layer, perm_idx)

    def set_mode(is_sparse=False):
        for i, layer in enumerate(model.model.layers):
            if i in sparse_up_projs:
                if is_sparse:
                    layer.mlp.up_proj = sparse_up_projs[i]
                else:
                    layer.mlp.up_proj = orig_up_projs[i]

    # Run Benchmark
    stats = {"Dense Win": 0, "Sparse Win": 0, "Tie": 0, "Both Bad": 0, "Error": 0}
    report_blocks = []

    print(f"\nStarting benchmark on {len(questions)} questions...\n")
    
    for q in questions:
        print(f"--- Q{q['id']} [{q['category']}] ---")
        prompt = q['prompt']
        
        # 1. Dense Run
        set_mode(is_sparse=False)
        t0 = time.time()
        dense_answer = generate(model, tokenizer, prompt=prompt, max_tokens=150, verbose=False)
        t_dense = time.time() - t0
        print(f"  Dense Done in {t_dense:.2f}s")
        
        # 2. Sparse Run
        t0 = time.time()
        sparse_answer = generate_custom(model, tokenizer, prompt=prompt, prefill_sparse=False, decode_sparse=True, set_mode_func=set_mode, max_tokens=150)
        t_sparse = time.time() - t0
        print(f"  Sparse Done in {t_sparse:.2f}s")
        
        # 3. Judge Evaluation
        print("  Calling Judge Service...")
        judge_result = judge_evaluate(prompt, dense_answer, sparse_answer)
        
        if judge_result:
            winner = judge_result.get("final_winner", "Error")
            # Map A -> Dense, B -> Sparse
            if winner == "A":
                winner_label = "Dense Win"
            elif winner == "B":
                winner_label = "Sparse Win"
            else:
                winner_label = winner
                
            if winner_label in stats:
                stats[winner_label] += 1
            else:
                stats["Error"] += 1
                
            print(f"  Result: {winner_label}\n")
            
            run_1 = judge_result.get("run_1", {})
            run_2 = judge_result.get("run_2", {})
            reasoning = f"**Run 1 (Dense vs Sparse):** {run_1.get('reasoning')}\n\n**Run 2 (Sparse vs Dense):** {run_2.get('reasoning')}"
        else:
            winner_label = "Error"
            stats["Error"] += 1
            reasoning = "Judge Service Failed or Offline."
            print(f"  Result: {winner_label}\n")

        # Format block
        block = f"## Q{q['id']} - [{q['category']}]\n\n"
        block += f"**Prompt:**\n> {prompt}\n\n"
        block += f"### Dense (满血版)\n{dense_answer}\n\n"
        block += f"### 1D Column Sparse (动态列稀疏版)\n{sparse_answer}\n\n"
        block += f"### 裁判判决: {winner_label}\n{reasoning}\n\n---\n"
        report_blocks.append(block)
        
        report = "# 1D 列稀疏架构 Benchmark 报告\n\n"
        report += "## 裁判胜率统计\n"
        report += "| Dense Win | Sparse Win | Tie | Both Bad | Error |\n"
        report += "|---|---|---|---|---|\n"
        report += f"| {stats['Dense Win']} | {stats['Sparse Win']} | {stats['Tie']} | {stats['Both Bad']} | {stats['Error']} |\n\n"
        report += "---\n\n"
        report += "".join(report_blocks)
        
        report_path = os.path.join(script_dir, "benchmark_report_1d_col.md")
        with open(report_path, "w") as f:
            f.write(report)
            
        sys.stdout.flush()
        
    print(f"Benchmark completed! Stats: {stats}")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
