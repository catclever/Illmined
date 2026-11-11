import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import time
import argparse
import sys
import os

from profiler import ProfileMLPWrapper
from pruner import prune_mlp_layer

def prefill_and_profile(model, tokenizer, prompt, target_layers):
    """
    Runs the prefill phase on the prompt, while recording activation magnitudes
    using the ProfileMLPWrapper.
    """
    prompt_tokens = mx.array([tokenizer.encode(prompt)])
    
    # Wrap target layers with profiler
    profilers = {}
    for i in target_layers:
        orig_mlp = model.model.layers[i].mlp
        profiler = ProfileMLPWrapper(orig_mlp)
        model.model.layers[i].mlp = profiler
        profilers[i] = profiler
        
    print(f"Running Prefill and Profiling (Prompt Length: {prompt_tokens.shape[1]})...")
    t0 = time.time()
    
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    
    # Run the model to populate KV cache and collect activations
    logits = model(prompt_tokens, cache=cache)
    next_token = mx.argmax(logits[:, -1, :], axis=-1)[None]
    
    # Await mlx evaluation to make sure everything computes
    mx.eval(logits, cache)
    
    prefill_time = time.time() - t0
    print(f"Prefill Completed in {prefill_time:.4f}s")
    
    return next_token, cache, profilers

def contextual_prune(model, profilers, target_layers, sparsity_ratio):
    """
    Collects scores from profilers, finds top K neurons, and physically prunes the MLPs.
    """
    print(f"Performing Contextual Pruning (Target Sparsity: {sparsity_ratio*100:.1f}%)...")
    t0 = time.time()
    
    for i in target_layers:
        profiler = profilers[i]
        scores = profiler.get_mean_scores() # shape: (intermediate_size,)
        
        # Determine number of neurons to keep
        intermediate_size = scores.shape[0]
        keep_k = int(intermediate_size * (1.0 - sparsity_ratio))
        
        # Find the top K indices
        # argsort sorts ascending, so we take the last keep_k elements
        sorted_indices = mx.argsort(scores)
        keep_indices = sorted_indices[-keep_k:]
        
        # Evaluate to ensure indices are realized before slicing weights
        mx.eval(keep_indices)
        
        # Unwrap the profiler and get the original mlp
        orig_mlp = profiler.orig_mlp
        
        # Prune the MLP
        pruned_mlp = prune_mlp_layer(orig_mlp, keep_indices)
        
        # Assign the pruned MLP back to the model
        model.model.layers[i].mlp = pruned_mlp
        
        # Explicitly delete the reference to orig_mlp in profiler to allow Garbage Collection
        del profiler.orig_mlp
        
    # Explicitly clear the profilers dict as well
    profilers.clear()
    
    # Crucial: Clear MLX memory cache to actually free up VRAM for generation
    mx.clear_cache()
    
    prune_time = time.time() - t0
    
    # 打印显存情况
    active_mem = mx.get_active_memory() / (1024**3)
    print(f"Pruning and GC Completed in {prune_time:.4f}s")
    print(f"-> 当前 MLX 占用显存 (Active Memory): {active_mem:.2f} GB")

def decode(model, tokenizer, next_token, cache, max_tokens=100):
    """
    Runs the standard decode loop. Since the model has been physically pruned,
    this runs faster and uses less VRAM automatically.
    """
    print(f"Starting Generation (Max Tokens: {max_tokens})...")
    generated_tokens = [next_token.item()]
    
    t0 = time.time()
    for i in range(max_tokens - 1):
        logits = model(next_token, cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)[None]
        generated_tokens.append(next_token.item())
        
        mx.eval(next_token)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    decode_time = time.time() - t0
    text = tokenizer.decode(generated_tokens)
    
    tokens_generated = len(generated_tokens)
    print(f"Generation Completed in {decode_time:.4f}s ({tokens_generated/decode_time:.2f} tok/s)")
    
    return text

def main():
    model_name = "Qwen/Qwen3-8b"
    print(f"Loading Model: {model_name}...")
    model, tokenizer = load(model_name)
    model.eval()
    
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers.")
    
    # Example prompt for Role Play / Long Context
    prompt = "请阅读下面这篇完全虚构的科幻报道，并回答文末的问题：\n\n在2142年，银河系边缘的“泽塔星”爆发了一场极其罕见的“晶体瘟疫”。这种瘟疫并不会感染碳基生物，而是专门腐蚀高度发达的量子计算机核心。据星际调查局（FIB）昨日公布的最新解密报告显示，这场瘟疫的源头并非自然产生，而是一个被称为“暗影之手”的极端黑客组织秘密投放的。\n为了应对这场史无前例的危机，泽塔星的首席科学家艾琳博士紧急研发了一种名为“光流盾”的防御协议。该协议的核心机制非常独特：它通过发射高频反向脉冲，在量子晶体表面形成一层极其致密的等离子隔离带，从而彻底阻断了瘟疫代码在物理层面的写入权限。经过长达三个月的全星系部署，“光流盾”最终大获成功，成功挽救了泽塔星上超过85%的核心主控服务器。\n\n问题：根据上述报道，这场“晶体瘟疫”的源头到底是什么？艾琳博士研发的“光流盾”防御协议的核心机制又是什么？\n回答：根据报道，"
    
    # We prune the middle-to-late layers where "reasoning/knowledge" routing happens heavily
    target_layers = list(range(10, 26))
    target_sparsity = 0.6 # Cut 60% of neurons
    
    # 打印初始显存
    initial_mem = mx.get_active_memory() / (1024**3)
    print(f"\n[显存追踪] 初始加载全模型占用显存: {initial_mem:.2f} GB")
    
    # 1. Prefill (Read) & Profile
    next_token, cache, profilers = prefill_and_profile(model, tokenizer, prompt, target_layers)
    
    prefill_mem = mx.get_active_memory() / (1024**3)
    peak_mem = mx.get_peak_memory() / (1024**3)
    print(f"[显存追踪] Prefill 结束后，剪枝前当前显存: {prefill_mem:.2f} GB (峰值: {peak_mem:.2f} GB)")
    
    # 2. Slice Weights & Free VRAM
    contextual_prune(model, profilers, target_layers, sparsity_ratio=target_sparsity)
    
    pruned_mem = mx.get_active_memory() / (1024**3)
    print(f"[显存追踪] 物理瘦身并清理 Cache 后显存: {pruned_mem:.2f} GB")
    print(f"[显存追踪] 恭喜！本次物理裁剪共释放了: {prefill_mem - pruned_mem:.2f} GB 显存！\n")
    
    # 3. Decode with lightweight model
    text = decode(model, tokenizer, next_token, cache, max_tokens=100)
    
    print("\n--- 最终生成结果 ---")
    print(text)
    print("-------------------")

if __name__ == "__main__":
    main()
