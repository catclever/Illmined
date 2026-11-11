import mlx.core as mx
import mlx.nn as nn

class MacMoE(nn.Module):
    def __init__(self, orig_mlp, expert_indices_list, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = orig_mlp.gate_proj.weight.shape[1]
        
        self.gate_projs = []
        self.up_projs = []
        self.down_projs = []
        
        centroids = []
        
        for e in range(num_experts):
            idx = mx.array(expert_indices_list[e])
            
            g_w = orig_mlp.gate_proj.weight[idx, :]
            u_w = orig_mlp.up_proj.weight[idx, :]
            d_w = orig_mlp.down_proj.weight[:, idx]
            
            g_proj = nn.Linear(self.hidden_size, len(idx), bias=False)
            g_proj.weight = g_w
            
            u_proj = nn.Linear(self.hidden_size, len(idx), bias=False)
            u_proj.weight = u_w
            
            d_proj = nn.Linear(len(idx), self.hidden_size, bias=False)
            d_proj.weight = d_w
            
            self.gate_projs.append(g_proj)
            self.up_projs.append(u_proj)
            self.down_projs.append(d_proj)
            
            # Heuristic: centroid of the gate features
            centroid = mx.mean(g_w, axis=0)
            centroids.append(centroid)
            
        # Router
        self.router = nn.Linear(self.hidden_size, num_experts, bias=False)
        self.router.weight = mx.stack(centroids, axis=0)
        
    def __call__(self, x):
        # x shape: [B, L, D]
        logits = self.router(x)
        
        # COARSE-GRAINED ROUTING: 
        # Average logits over the sequence (L) to pick the best experts for this entire context.
        # This completely avoids slow token-by-token sparse indexing.
        seq_logits = mx.mean(logits, axis=(0, 1)) 
        
        # Get Top-K experts
        top_k_indices = mx.argsort(seq_logits)[-self.top_k:]
        mx.eval(top_k_indices)
        top_k_list = top_k_indices.tolist()
        
        out = mx.zeros_like(x)
        
        for e in top_k_list:
            gate = self.gate_projs[e](x)
            up = self.up_projs[e](x)
            act = nn.silu(gate) * up
            down = self.down_projs[e](act)
            out = out + down
            
        return out
