import time
import os
import pickle
import mlx.core as mx
from mlx_lm import load, generate
from moe_layer import MacMoE

def main():
    model_name = "Qwen/Qwen3-8b"
    print(f"Loading base model {model_name}...")
    try:
        model, tokenizer = load(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}. Attempting fallback...")
        model_name = "Qwen/Qwen2.5-7B"
        model, tokenizer = load(model_name)
        
    model.eval()

    clusters_path = "clustered_moe/outputs/neuron_clusters.pkl"
    if not os.path.exists(clusters_path):
        print(f"Error: {clusters_path} not found. Please run neuron_clustering.py first.")
        return

    with open(clusters_path, 'rb') as f:
        clusters = pickle.load(f)

    target_layers = list(range(10, 26))
    num_experts = 8
    top_k = 8

    print(f"Replacing layers {target_layers[0]} to {target_layers[-1]} with Clustered MoE...")
    for i in target_layers:
        if i in clusters and i < len(model.model.layers):
            orig_mlp = model.model.layers[i].mlp
            moe_mlp = MacMoE(orig_mlp, clusters[i], num_experts=num_experts, top_k=top_k)
            model.model.layers[i].mlp = moe_mlp

    # Force evaluation of weights
    mx.eval(model.parameters())

    prompt = """请阅读下面这篇完全虚构的科幻报道，并回答文末的问题：

在2142年，银河系边缘的“泽塔星”爆发了一场极其罕见的“晶体瘟疫”。这种瘟疫并不会感染碳基生物，而是专门腐蚀高度发达的量子计算机核心。据星际调查局（FIB）昨日公布的最新解密报告显示，这场瘟疫的源头并非自然产生，而是一个被称为“暗影之手”的极端黑客组织秘密投放的。
为了应对这场史无前例的危机，泽塔星的首席科学家艾琳博士紧急研发了一种名为“光流盾”的防御协议。该协议的核心机制非常独特：它通过发射高频反向脉冲，在量子晶体表面形成一层极其致密的等离子隔离带，从而彻底阻断了瘟疫代码在物理层面的写入权限。经过长达三个月的全星系部署，“光流盾”最终大获成功，成功挽救了泽塔星上超过85%的核心主控服务器。

问题：艾琳博士研发的“光流盾”是如何起作用的？它的核心防御机制是什么？"""

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print("\nStarting generation with Clustered MoE Router...")
    print("-" * 50)
    
    from mlx_lm.models.cache import make_prompt_cache
    
    def generate_step(prompt, model):
        tokens = tokenizer.encode(prompt)
        tokens = mx.array(tokens)[None]
        
        cache = make_prompt_cache(model)
        
        # Prefill
        tic = time.perf_counter()
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        prefill_time = time.perf_counter() - tic
        
        next_token = mx.argmax(logits[:, -1, :], axis=-1)[None]
        generated = [next_token.item()]
        
        # Decode
        tic = time.perf_counter()
        for _ in range(200):
            logits = model(next_token, cache=cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)[None]
            generated.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break
        mx.eval(logits)
        decode_time = time.perf_counter() - tic
        
        out_text = tokenizer.decode(generated)
        print(out_text)
        print("-" * 50)
        print(f"Prefill speed: {tokens.shape[1] / prefill_time:.2f} tok/s")
        print(f"Decode speed: {len(generated) / decode_time:.2f} tok/s")

    generate_step(prompt_text, model)

if __name__ == "__main__":
    main()
