import os
import json
import pickle
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from sklearn.cluster import KMeans
from tqdm import tqdm

def load_calibration_data(jsonl_path, num_samples=30):
    dataset = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Support both MT-Bench format and HellaSwag format
            if 'turns' in data:
                dataset.append(data['turns'][0])
            elif 'ctx' in data:
                dataset.append(data['ctx'])
            if len(dataset) >= num_samples:
                break
    return dataset

def balanced_kmeans_assignment(X, num_clusters, cluster_size):
    """
    X: [num_features, num_samples]
    Assign each feature to exactly one cluster, such that each cluster has `cluster_size` features.
    """
    print(f"Running KMeans with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    
    print("Performing balanced assignment...")
    # Calculate distance from each feature to each cluster center
    # X: (N, D), centers: (K, D)
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2) # (N, K)
    
    # Greedy assignment
    assignment = np.full(X.shape[0], -1, dtype=int)
    cluster_counts = np.zeros(num_clusters, dtype=int)
    
    # Create a list of all (feature_idx, cluster_idx, distance)
    feature_indices, cluster_indices = np.indices(distances.shape)
    flat_distances = distances.flatten()
    flat_features = feature_indices.flatten()
    flat_clusters = cluster_indices.flatten()
    
    # Sort by distance
    sort_idx = np.argsort(flat_distances)
    
    assigned_count = 0
    for idx in sort_idx:
        feat = flat_features[idx]
        clust = flat_clusters[idx]
        
        if assignment[feat] == -1 and cluster_counts[clust] < cluster_size:
            assignment[feat] = clust
            cluster_counts[clust] += 1
            assigned_count += 1
            
        if assigned_count == X.shape[0]:
            break
            
    # Generate permutation index array
    # We want features in cluster 0 first, then cluster 1, etc.
    perm_idx = np.argsort(assignment)
    return perm_idx

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "Qwen/Qwen3-8b"
    
    print("Loading model...")
    model, tokenizer = load(model_path)
    model.eval()
    
    # We will hook into the layers to get the input to up_proj
    target_layers = list(range(len(model.model.layers)))
    
    # Prepare a dictionary to store activations per layer
    # We will accumulate the absolute values of activations
    layer_activations = {i: [] for i in target_layers}
    
    # Install hooks
    original_forwards = {}
    
    class HookedLinear(nn.Module):
        def __init__(self, orig_layer, layer_idx):
            super().__init__()
            self.orig_layer = orig_layer
            self.layer_idx = layer_idx
            # We must set weight so any attribute access works
            self.weight = orig_layer.weight
            if hasattr(orig_layer, 'bias'):
                self.bias = orig_layer.bias
                
        def __call__(self, x):
            # Capture x
            B, L, D_in = x.shape
            x_flat = mx.abs(x).reshape(-1, D_in)
            step = max(1, x_flat.shape[0] // 128)
            x_sub = x_flat[::step, :]
            if x_sub.shape[0] > 128:
                x_sub = x_sub[:128, :]
            mx.eval(x_sub)
            layer_activations[self.layer_idx].append(np.array(x_sub.astype(mx.float32)))
            return self.orig_layer(x)

    print("Installing hooks...")
    for i in target_layers:
        original_forwards[i] = model.model.layers[i].mlp.up_proj
        model.model.layers[i].mlp.up_proj = HookedLinear(model.model.layers[i].mlp.up_proj, i)

    # Load Calibration Data
    calib_path = os.path.join(script_dir, "../dynamic_sparsity/benchmark/data/hellaswag_val.jsonl")
    if not os.path.exists(calib_path):
        calib_path = os.path.join(script_dir, "../dynamic_sparsity/benchmark/question.jsonl")
        
    print(f"Loading calibration data from {calib_path}")
    dataset = load_calibration_data(calib_path, num_samples=64)
    
    print("Running calibration forward passes...")
    for text in tqdm(dataset):
        tokens = tokenizer.encode(text)
        if len(tokens) > 512:
            tokens = tokens[:512]
        x = mx.array([tokens])
        _ = model(x)
        
    # Restore original forwards
    for i in target_layers:
        model.model.layers[i].mlp.up_proj = original_forwards[i]
        
    print("Processing activations and generating permutations...")
    permutations = {}
    
    D_in = model.model.layers[0].mlp.up_proj.weight.shape[1]
    block_size = 64
    num_clusters = D_in // block_size
    
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    for i in tqdm(target_layers):
        # Concatenate all collected activations for this layer
        # shape: (num_tokens_total, D_in)
        act = np.concatenate(layer_activations[i], axis=0) 
        
        # Transpose to (D_in, num_tokens_total)
        # Each feature is a point in `num_tokens_total` dimensional space
        X = act.T
        
        # Standardize features (optional, but helps clustering focus on shape rather than magnitude)
        # Actually, magnitude is important! A feature that is always 0 shouldn't be clustered with a highly active feature.
        # But weighting by weight norm is also good.
        # Let's multiply by W_col_norm to represent the actual IMPACT.
        W = np.array(model.model.layers[i].mlp.up_proj.weight.astype(mx.float32))
        W_col_norm = np.sum(np.abs(W), axis=0) # (D_in,)
        
        X_impact = X * W_col_norm[:, None]
        
        perm_idx = balanced_kmeans_assignment(X_impact, num_clusters, block_size)
        permutations[i] = perm_idx.tolist()
        
    perm_path = os.path.join(out_dir, "qwen_1d_col_cluster_perm.pkl")
    with open(perm_path, "wb") as f:
        pickle.dump(permutations, f)
        
    print(f"Permutations saved to {perm_path}")

if __name__ == "__main__":
    main()
