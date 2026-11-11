import mlx.core as mx
import mlx.nn as nn

def prune_mlp_layer(mlp, keep_indices):
    """
    Physically slices the weight matrices of the MLP layer to keep only the active neurons.
    This permanently reduces the VRAM and compute requirements of the layer.
    
    Args:
        mlp: The original MLP module (e.g. Qwen2MLP)
        keep_indices: An mx.array containing the indices of the intermediate neurons to keep.
    """
    # In MLX, nn.Linear weights are stored as (out_features, in_features)
    # gate_proj maps from hidden_size to intermediate_size, so shape is (intermediate_size, hidden_size)
    # up_proj maps from hidden_size to intermediate_size, so shape is (intermediate_size, hidden_size)
    # down_proj maps from intermediate_size to hidden_size, so shape is (hidden_size, intermediate_size)

    # 1. Prune gate_proj
    original_gate_weight = mlp.gate_proj.weight
    # Adding 0 forces MLX to allocate a new tensor instead of returning a view of the original large buffer
    new_gate_weight = original_gate_weight[keep_indices, :] + 0.0
    new_gate_proj = nn.Linear(new_gate_weight.shape[1], new_gate_weight.shape[0], bias=False)
    new_gate_proj.weight = new_gate_weight
    mlp.gate_proj = new_gate_proj
    
    # 2. Prune up_proj
    original_up_weight = mlp.up_proj.weight
    new_up_weight = original_up_weight[keep_indices, :] + 0.0
    new_up_proj = nn.Linear(new_up_weight.shape[1], new_up_weight.shape[0], bias=False)
    new_up_proj.weight = new_up_weight
    mlp.up_proj = new_up_proj
    
    # 3. Prune down_proj
    original_down_weight = mlp.down_proj.weight
    new_down_weight = original_down_weight[:, keep_indices] + 0.0
    new_down_proj = nn.Linear(new_down_weight.shape[1], new_down_weight.shape[0], bias=False)
    new_down_proj.weight = new_down_weight
    mlp.down_proj = new_down_proj
    
    return mlp
