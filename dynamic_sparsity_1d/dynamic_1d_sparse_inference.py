import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import time
import pickle
import os

class PermutedSparseLinear(nn.Module):
    def __init__(self, sparse_linear, perm_idx):
        super().__init__()
        self.sl = sparse_linear
        self.perm_idx = perm_idx
        
    def __call__(self, x):
        return self.sl(x[..., self.perm_idx])

class DynamicColumnSparseLinear(nn.Module):
    """
    1D Column Sparsity (Sparsifying the D_in dimension).
    Drops input features based on W_col_norm and x_norm dynamically.
    """
    def __init__(self, orig_layer, sparsity_ratio):
        super().__init__()
        self.weight = orig_layer.weight
        self.sparsity_ratio = sparsity_ratio
        
        D_out, D_in = self.weight.shape
        self.B_in = 64
        self.N_in = D_in // self.B_in
        
        W_blocks = self.weight.reshape((D_out, self.N_in, self.B_in))
        self.W_col_norm = mx.sum(mx.abs(W_blocks), axis=(0, 2))
        
    def __call__(self, x):
        B, L, D_in = x.shape
        D_out, _ = self.weight.shape
        
        x_blocks = x.reshape((B, L, self.N_in, self.B_in))
        x_norm = mx.sum(mx.abs(x_blocks), axis=-1)
        
        Impact = x_norm * self.W_col_norm.reshape((1, 1, self.N_in))
        
        sorted_impact = mx.sort(Impact, axis=-1)
        idx = int(self.sparsity_ratio * self.N_in)
        idx = min(max(idx, 0), self.N_in - 1)
        threshold_val = sorted_impact[:, :, idx:idx+1]
        
        Mask = Impact > threshold_val
        
        Mask_expanded = Mask.reshape((B, L, self.N_in, 1))
        Mask_expanded = mx.broadcast_to(Mask_expanded, (B, L, self.N_in, self.B_in))
        Mask_expanded = Mask_expanded.reshape((B, L, 1, D_in))
        
        x_expanded = x.reshape((B, L, 1, D_in))
        W_expanded = self.weight.reshape((1, 1, D_out, D_in))
        
        masked_W = W_expanded * Mask_expanded
        Z = mx.sum(x_expanded * masked_W, axis=-1)
        return Z

class DynamicRowSparseLinear(nn.Module):
    """
    1D Row Sparsity (Sparsifying the D_out dimension).
    Drops output neurons based on interaction between x_norm and W_norm blocks dynamically.
    """
    def __init__(self, orig_layer, sparsity_ratio):
        super().__init__()
        self.weight = orig_layer.weight
        self.sparsity_ratio = sparsity_ratio
        
        D_out, D_in = self.weight.shape
        self.B_in, self.B_out = 64, 64
        self.N_in, self.N_out = D_in // self.B_in, D_out // self.B_out
        
        W_blocks = self.weight.reshape((self.N_out, self.B_out, self.N_in, self.B_in))
        self.W_norm = mx.sum(mx.abs(W_blocks), axis=(1, 3))
        
    def __call__(self, x):
        B, L, D_in = x.shape
        D_out, _ = self.weight.shape
        
        x_blocks = x.reshape((B, L, self.N_in, self.B_in))
        x_norm = mx.sum(mx.abs(x_blocks), axis=-1)
        
        Impact = mx.sum(
            self.W_norm.reshape((1, 1, self.N_out, self.N_in)) * x_norm.reshape((B, L, 1, self.N_in)),
            axis=-1
        )
        
        sorted_impact = mx.sort(Impact, axis=-1)
        idx = int(self.sparsity_ratio * self.N_out)
        idx = min(max(idx, 0), self.N_out - 1)
        threshold_val = sorted_impact[:, :, idx:idx+1]
        
        Mask = Impact > threshold_val
        
        Mask_expanded = Mask.reshape((B, L, self.N_out, 1))
        Mask_expanded = mx.broadcast_to(Mask_expanded, (B, L, self.N_out, self.B_out))
        Mask_expanded = Mask_expanded.reshape((B, L, D_out, 1))
        
        x_expanded = x.reshape((B, L, 1, D_in))
        W_expanded = self.weight.reshape((1, 1, D_out, D_in))
        
        masked_W = W_expanded * Mask_expanded
        Z = mx.sum(x_expanded * masked_W, axis=-1)
        return Z

def generate_custom(model, tokenizer, prompt, prefill_sparse=False, decode_sparse=False, set_mode_func=None, max_tokens=50):
    prompt_tokens = mx.array([tokenizer.encode(prompt)])
    
    if set_mode_func:
        set_mode_func(is_sparse=prefill_sparse)
        
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(prompt_tokens, cache=cache)
    next_token = mx.argmax(logits[:, -1, :], axis=-1)[None]
    
    generated_tokens = [next_token.item()]
    
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
    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-8b")
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perm_path = os.path.join(script_dir, "outputs/qwen_1d_col_cluster_perm.pkl")
    with open(perm_path, 'rb') as f:
        permutation_maps = pickle.load(f)

    num_layers = len(model.model.layers)
    target_layers = list(range(10, 36))
    
    layer_sparsity = [0.0] * num_layers
    for idx, layer_idx in enumerate(target_layers):
        progress = idx / (len(target_layers) - 1)
        layer_sparsity[layer_idx] = 0.1 + progress * 0.6
    
    orig_up_projs = {}
    for i in range(num_layers):
        orig_up_projs[i] = model.model.layers[i].mlp.up_proj
        
    def setup_sparse_model(SparseClass):
        sparse_up_projs = {}
        for i in range(num_layers):
            if i in target_layers and i < len(permutation_maps):
                sparsity = layer_sparsity[i]
                perm_idx = mx.array(permutation_maps[i])
                shuffled_weight = mx.array(orig_up_projs[i].weight[:, perm_idx])
                
                temp_linear = nn.Linear(shuffled_weight.shape[1], shuffled_weight.shape[0], bias=False)
                temp_linear.weight = shuffled_weight
                
                sparse_layer = SparseClass(temp_linear, sparsity_ratio=sparsity)
                sparse_up_projs[i] = PermutedSparseLinear(sparse_layer, perm_idx)
        return sparse_up_projs

    print("Initializing sparse layers...")
    col_sparse_up_projs = setup_sparse_model(DynamicColumnSparseLinear)
    row_sparse_up_projs = setup_sparse_model(DynamicRowSparseLinear)

    def set_mode(mode="dense"):
        for i in range(num_layers):
            if mode == "col_sparse" and i in col_sparse_up_projs:
                model.model.layers[i].mlp.up_proj = col_sparse_up_projs[i]
            elif mode == "row_sparse" and i in row_sparse_up_projs:
                model.model.layers[i].mlp.up_proj = row_sparse_up_projs[i]
            else:
                model.model.layers[i].mlp.up_proj = orig_up_projs[i]

    prompt = """请阅读下面这篇完全虚构的科幻报道，并回答文末的问题：

在2142年，银河系边缘的“泽塔星”爆发了一场极其罕见的“晶体瘟疫”。这种瘟疫并不会感染碳基生物，而是专门腐蚀高度发达的量子计算机核心。据星际调查局（FIB）昨日公布的最新解密报告显示，这场瘟疫的源头并非自然产生，而是一个被称为“暗影之手”的极端黑客组织秘密投放的。
为了应对这场史无前例的危机，泽塔星的首席科学家艾琳博士紧急研发了一种名为“光流盾”的防御协议。该协议的核心机制非常独特：它通过发射高频反向脉冲，在量子晶体表面形成一层极其致密的等离子隔离带，从而彻底阻断了瘟疫代码在物理层面的写入权限。经过长达三个月的全星系部署，“光流盾”最终大获成功，成功挽救了泽塔星上超过85%的核心主控服务器。

问题：根据上述报道，这场“晶体瘟疫”的源头到底是什么？艾琳博士研发的“光流盾”防御协议的核心机制又是什么？
回答：根据报道，"""

    print("\n--- 1. 读题满血, 答题满血 (Prefill Dense, Decode Dense) ---")
    t0 = time.time()
    text = generate_custom(model, tokenizer, prompt, prefill_sparse=False, decode_sparse=False, set_mode_func=lambda is_sparse: set_mode("dense"))
    print(f"==========\n{text}\n==========\n耗时: {time.time()-t0:.4f}s")

    print("\n--- 2. 读题满血, 答题 1D 列稀疏 (Prefill Dense, Decode Col Sparse) ---")
    t0 = time.time()
    text = generate_custom(model, tokenizer, prompt, prefill_sparse=False, decode_sparse=True, set_mode_func=lambda is_sparse: set_mode("col_sparse" if is_sparse else "dense"))
    print(f"==========\n{text}\n==========\n耗时: {time.time()-t0:.4f}s")
    
    print("\n--- 3. 读题满血, 答题 1D 行稀疏 (Prefill Dense, Decode Row Sparse) ---")
    t0 = time.time()
    text = generate_custom(model, tokenizer, prompt, prefill_sparse=False, decode_sparse=True, set_mode_func=lambda is_sparse: set_mode("row_sparse" if is_sparse else "dense"))
    print(f"==========\n{text}\n==========\n耗时: {time.time()-t0:.4f}s")

if __name__ == "__main__":
    main()
