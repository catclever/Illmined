import mlx.core as mx
import mlx.nn as nn

class ProfileMLPWrapper(nn.Module):
    """
    A wrapper for the MLP layer that records the activation magnitude
    during the prefill (reading) phase.
    """
    def __init__(self, orig_mlp):
        super().__init__()
        self.orig_mlp = orig_mlp
        self.activation_scores = None
        self.token_count = 0

    def __call__(self, x):
        # Forward pass through gate and up projections
        gate_out = self.orig_mlp.gate_proj(x)
        up_out = self.orig_mlp.up_proj(x)
        
        # Calculate the activation that goes into the down_proj
        # Standard LLaMA / Qwen MLP activation: silu(gate) * up
        act = nn.silu(gate_out) * up_out
        
        # Accumulate absolute activation scores
        # act shape: (batch_size, sequence_length, intermediate_size)
        act_abs = mx.abs(act)
        sum_act = mx.sum(act_abs, axis=(0, 1))
        
        if self.activation_scores is None:
            self.activation_scores = sum_act
        else:
            self.activation_scores += sum_act
            
        self.token_count += x.shape[0] * x.shape[1]
        
        # Complete the forward pass
        return self.orig_mlp.down_proj(act)

    def get_mean_scores(self):
        """
        Returns the mean absolute activation for each intermediate neuron.
        """
        if self.activation_scores is None or self.token_count == 0:
            return None
        return self.activation_scores / self.token_count
