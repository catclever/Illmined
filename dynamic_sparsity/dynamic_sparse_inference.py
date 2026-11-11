import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import time
import pickle
import argparse
import sys
"""
dynamic_sparse_inference.py

核心推理框架模块。
包含稀疏化核心算法的定义：
- StaticSparseLinear: 静态按权重大小进行稀疏化的线性层
- PermutedSparseLinear: 处理矩阵重排（为了更好地分块）的包裹层
- DynamicAbsoluteSparseLinear: 基于输入激活 (Activation) 与权重 (Weight) 动态计算注意力/重要性，
  并利用 argsort 动态进行按需稀疏化的核心创新层。

注意：由于 DynamicAbsoluteSparseLinear 在每次推理时都需要根据输入动态计算掩码并进行排序 (argsort)，
如果直接在 Python/MLX 层面不加 custom kernel 运行，速度会相对较慢，但可用于验证算法正确性。
"""
import os

class PermutedSparseLinear(nn.Module):
    def __init__(self, sparse_linear, perm_idx):
        super().__init__()
        self.sl = sparse_linear
        self.perm_idx = perm_idx
        
    def __call__(self, x):
        return self.sl(x[..., self.perm_idx])

class StaticSparseLinear(nn.Module):
    def __init__(self, orig_layer, sparsity_ratio):
        super().__init__()
        self.weight = orig_layer.weight
        self.sparsity_ratio = sparsity_ratio
        
        D_out, D_in = self.weight.shape
        B_in, B_out = 64, 64
        N_in, N_out = D_in // B_in, D_out // B_out
        
        W_blocks = self.weight.reshape((N_out, B_out, N_in, B_in))
        W_norm = mx.sum(mx.abs(W_blocks), axis=(1, 3))
        
        flat_impact = W_norm.reshape(-1)
        sorted_impact = mx.sort(flat_impact)
        idx = int(sparsity_ratio * flat_impact.size)
        idx = min(max(idx, 0), flat_impact.size - 1)
        threshold_val = sorted_impact[idx]
        Mask = W_norm > threshold_val
        
        Mask_expanded = Mask.reshape((N_out, 1, N_in, 1))
        Mask_expanded = mx.broadcast_to(Mask_expanded, (N_out, B_out, N_in, B_in))
        self.Mask_expanded = Mask_expanded.reshape((D_out, D_in))
        
        self.sparse_weight = self.weight * self.Mask_expanded
        
    def update_weight(self, new_weight):
        self.weight = new_weight
        self.sparse_weight = self.weight * self.Mask_expanded
        
    def __call__(self, x):
        return x @ self.sparse_weight.T

class DynamicAbsoluteSparseLinear(nn.Module):
    def __init__(self, orig_layer, sparsity_ratio):
        super().__init__()
        self.weight = orig_layer.weight
        self.sparsity_ratio = sparsity_ratio
        
        D_out, D_in = self.weight.shape
        self.B_in, self.B_out = 64, 64
        self.N_in, self.N_out = D_in // self.B_in, D_out // self.B_out
        
        W_blocks = self.weight.reshape((self.N_out, self.B_out, self.N_in, self.B_in))
        self.W_norm = mx.sum(mx.abs(W_blocks), axis=(1, 3))
        
        # Compile Custom Metal Kernel
        source = f"""
        uint D_in = {D_in};
        uint N_in = {self.N_in};
        uint B_in = {self.B_in};
        uint B_out = {self.B_out};
        float sparsity_ratio = {self.sparsity_ratio};
        
        uint out_idx = thread_position_in_grid.x;
        uint block_out = out_idx / B_out;
        uint local_id = thread_position_in_threadgroup.x; // 0 to 63
        
        // Parallel Bitonic Sort in Shared Memory (N=64)
        threadgroup float shared_impacts[64];
        
        if (local_id < N_in) {{
            shared_impacts[local_id] = w_norm[block_out * N_in + local_id] * x_norm[local_id];
        }} else {{
            shared_impacts[local_id] = -1e9;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Bitonic Sort for 64 elements (21 steps)
        for (uint k = 2; k <= 64; k *= 2) {{
            for (uint j = k / 2; j > 0; j /= 2) {{
                uint ixj = local_id ^ j;
                if (ixj > local_id) {{
                    if ((local_id & k) == 0) {{
                        // Ascending
                        if (shared_impacts[local_id] > shared_impacts[ixj]) {{
                            float tmp = shared_impacts[local_id];
                            shared_impacts[local_id] = shared_impacts[ixj];
                            shared_impacts[ixj] = tmp;
                        }}
                    }} else {{
                        // Descending
                        if (shared_impacts[local_id] < shared_impacts[ixj]) {{
                            float tmp = shared_impacts[local_id];
                            shared_impacts[local_id] = shared_impacts[ixj];
                            shared_impacts[ixj] = tmp;
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }}
        
        // Exact percentile threshold
        uint threshold_idx = (uint)(sparsity_ratio * 64.0);
        if (threshold_idx >= 64) threshold_idx = 63;
        float threshold = shared_impacts[threshold_idx];
        
        float sum = 0.0;
        for (uint block_in = 0; block_in < N_in; ++block_in) {{
            float original_impact = w_norm[block_out * N_in + block_in] * x_norm[block_in];
            if (original_impact >= threshold) {{
                for (uint j = 0; j < B_in; ++j) {{
                    uint in_idx = block_in * B_in + j;
                    sum += static_cast<float>(x[in_idx]) * static_cast<float>(w[out_idx * D_in + in_idx]);
                }}
            }}
        }}
        out[out_idx] = static_cast<__typeof__(out[0])>(sum);
        """
        self.kernel = mx.fast.metal_kernel(
            name=f"dynamic_sparse_gemm_{D_in}_{D_out}",
            input_names=["x", "w", "w_norm", "x_norm"],
            output_names=["out"],
            source=source
        )
        
    def __call__(self, x):
        B, L, D_in = x.shape
        D_out, _ = self.weight.shape
        
        # Fallback to dense for prefill to avoid OOM or slow python loops
        if L > 1 or B > 1:
            return x @ self.weight.T
            
        x_blocks = x.reshape((self.N_in, self.B_in))
        x_norm = mx.sum(mx.abs(x_blocks), axis=-1)
        
        # Ensure all arrays are flat and contiguous before passing to Metal
        x_flat = mx.reshape(x, (-1,))
        w_flat = mx.reshape(self.weight, (-1,))
        w_norm_flat = mx.reshape(self.W_norm, (-1,))
        x_norm_flat = mx.reshape(x_norm, (-1,))
        
        # Run Custom Metal Kernel
        outputs = self.kernel(
            inputs=[x_flat, w_flat, w_norm_flat, x_norm_flat],
            template=[],
            grid=(D_out, 1, 1),
            threadgroup=(self.B_out, 1, 1),
            output_shapes=[(D_out,)],
            output_dtypes=[x.dtype]
        )
        Z = outputs[0]
        return Z.reshape((B, L, D_out))

def generate_custom(model, tokenizer, prompt, prefill_sparse=False, decode_sparse=False, set_mode_func=None, max_tokens=50):
    prompt_tokens = mx.array([tokenizer.encode(prompt)])
    
    # --- Prefill ---
    if set_mode_func:
        set_mode_func(is_sparse=prefill_sparse)
        
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(prompt_tokens, cache=cache)
    next_token = mx.argmax(logits[:, -1, :], axis=-1)[None]
    
    generated_tokens = [next_token.item()]
    
    # --- Decode ---
    if set_mode_func:
        set_mode_func(is_sparse=decode_sparse)
        
    for _ in range(max_tokens - 1):
        logits = model(next_token, cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)[None]
        generated_tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated_tokens)

def main():
    model, tokenizer = load("Qwen/Qwen3-8b")
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perm_path = os.path.join(script_dir, "../weight_clustering/outputs/qwen_all_layers_permutation.pkl")
    with open(perm_path, 'rb') as f:
        permutation_maps = pickle.load(f)

    num_layers = len(model.model.layers)
    target_layers = list(range(10, 36))
    
    layer_sparsity = [0.0] * num_layers
    for idx, layer_idx in enumerate(target_layers):
        progress = idx / (len(target_layers) - 1)
        layer_sparsity[layer_idx] = 0.1 + progress * 0.6
    
    sparse_up_projs = {}
    orig_up_projs = {}
    
    for i in range(num_layers):
        layer = model.model.layers[i]
        orig_up_projs[i] = layer.mlp.up_proj
        
        if i in target_layers and i < len(permutation_maps):
            sparsity = layer_sparsity[i]
            perm_idx = mx.array(permutation_maps[i])
            shuffled_weight = mx.array(orig_up_projs[i].weight[:, perm_idx])
            
            temp_linear = nn.Linear(shuffled_weight.shape[1], shuffled_weight.shape[0], bias=False)
            temp_linear.weight = shuffled_weight
            
            sparse_layer = StaticSparseLinear(temp_linear, sparsity_ratio=sparsity)
            sparse_up_projs[i] = PermutedSparseLinear(sparse_layer, perm_idx)

    def set_mode(is_sparse=False):
        for i in range(num_layers):
            if is_sparse and i in sparse_up_projs:
                model.model.layers[i].mlp.up_proj = sparse_up_projs[i]
            else:
                model.model.layers[i].mlp.up_proj = orig_up_projs[i]

    prompt = """请阅读下面这篇完全虚构的科幻报道，并回答文末的问题：

在2142年，银河系边缘的“泽塔星”爆发了一场极其罕见的“晶体瘟疫”。这种瘟疫并不会感染碳基生物，而是专门腐蚀高度发达的量子计算机核心。据星际调查局（FIB）昨日公布的最新解密报告显示，这场瘟疫的源头并非自然产生，而是一个被称为“暗影之手”的极端黑客组织秘密投放的。
为了应对这场史无前例的危机，泽塔星的首席科学家艾琳博士紧急研发了一种名为“光流盾”的防御协议。该协议的核心机制非常独特：它通过发射高频反向脉冲，在量子晶体表面形成一层极其致密的等离子隔离带，从而彻底阻断了瘟疫代码在物理层面的写入权限。经过长达三个月的全星系部署，“光流盾”最终大获成功，成功挽救了泽塔星上超过85%的核心主控服务器。

问题：根据上述报道，这场“晶体瘟疫”的源头到底是什么？艾琳博士研发的“光流盾”防御协议的核心机制又是什么？
回答：根据报道，"""

    print("\n--- 1. 读题满血, 答题满血 (Prefill Dense, Decode Dense) ---")
    t0 = time.time()
    text = generate_custom(model, tokenizer, prompt, prefill_sparse=False, decode_sparse=False, set_mode_func=set_mode)
    print(f"==========\n{text}\n==========\n耗时: {time.time()-t0:.4f}s")

    print("\n--- 2. 读题满血, 答题残血 (Prefill Dense, Decode Untrained Sparse) ---")
    t0 = time.time()
    text = generate_custom(model, tokenizer, prompt, prefill_sparse=False, decode_sparse=True, set_mode_func=set_mode)
    print(f"==========\n{text}\n==========\n耗时: {time.time()-t0:.4f}s")

    # We skip test case 3 (蒸馏残血) since the training approach was mathematically proven flawed for generation.
    # Instead, we directly compare Static Sparse (baseline) vs Dynamic Absolute Sparse (our optimal solution).
    
    # Switch to Dynamic Absolute Sparsity
    dynamic_sparse_up_projs = {}
    for i in range(num_layers):
        if i in target_layers and i < len(permutation_maps):
            sparsity = layer_sparsity[i]
            perm_idx = mx.array(permutation_maps[i])
            shuffled_weight = mx.array(orig_up_projs[i].weight[:, perm_idx])
            
            temp_linear = nn.Linear(shuffled_weight.shape[1], shuffled_weight.shape[0], bias=False)
            temp_linear.weight = shuffled_weight
            
            sparse_layer = DynamicAbsoluteSparseLinear(temp_linear, sparsity_ratio=sparsity)
            dynamic_sparse_up_projs[i] = PermutedSparseLinear(sparse_layer, perm_idx)
            
    def set_dynamic_mode(is_sparse=False):
        for i in range(num_layers):
            if is_sparse and i in dynamic_sparse_up_projs:
                model.model.layers[i].mlp.up_proj = dynamic_sparse_up_projs[i]
            else:
                model.model.layers[i].mlp.up_proj = orig_up_projs[i]
                
    print("\n--- 3. 读题满血, 答题【动态绝对】残血 (Prefill Dense, Decode Dynamic Absolute Sparse) ---")
    t0 = time.time()
    text = generate_custom(model, tokenizer, prompt, prefill_sparse=False, decode_sparse=True, set_mode_func=set_dynamic_mode)
    print(f"==========\n{text}\n==========\n耗时: {time.time()-t0:.4f}s")

if __name__ == "__main__":
    main()
