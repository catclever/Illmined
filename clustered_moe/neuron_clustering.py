import mlx.core as mx
import os
import pickle
from mlx_lm import load

def cluster_neurons(model_name="Qwen/Qwen3-8b", target_layers=range(10, 26), num_experts=8):
    print(f"Loading {model_name} for neuron clustering...")
    model, _ = load(model_name)
    
    os.makedirs("clustered_moe/outputs", exist_ok=True)
    out_path = "clustered_moe/outputs/neuron_clusters.pkl"
    
    clusters = {}
    for i in target_layers:
        print(f"Clustering layer {i} into {num_experts} experts...")
        w = model.model.layers[i].mlp.gate_proj.weight
        
        # Calculate L2 norm of each neuron to group similar magnitude neurons together
        norms = mx.linalg.norm(w, axis=1)
        
        # Sort neurons by norm (this gives a balanced 1D projection clustering)
        sorted_indices = mx.argsort(norms)
        # Evaluate to realize the array
        mx.eval(sorted_indices)
        sorted_indices_list = sorted_indices.tolist()
        
        # Split into `num_experts` balanced chunks
        intermediate_size = w.shape[0]
        expert_size = intermediate_size // num_experts
        
        layer_clusters = []
        for e in range(num_experts):
            start = e * expert_size
            end = (e + 1) * expert_size if e < num_experts - 1 else intermediate_size
            expert_indices = sorted_indices_list[start:end]
            layer_clusters.append(expert_indices)
            
        clusters[i] = layer_clusters
        
    with open(out_path, 'wb') as f:
        pickle.dump(clusters, f)
        
    print(f"Clustering complete. Saved to {out_path}")

if __name__ == "__main__":
    cluster_neurons()
