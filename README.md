# 虚室生白 (Illmined)

基于 **Apple Silicon 统一内存架构（UMA, MLX + Metal）** 的端侧大语言模型 **免训练动态稀疏（Dynamic Sparsity）与非对称推理加速** 探索仓库。

## 算力与功耗难题
在端侧部署大语言模型，传统的全量参数前向传播（Dense GEMM）在每一次生成新词时，都会迫使硬件执行庞大的全连接矩阵乘法。这不仅严重占用统一内存读取带宽，更会带来显著的设备发热与功耗问题。

本仓库的核心思路非常直接：**在推断生成时，动态切断大模型中海量无用的全连接（MLP）计算**——让硬件具备“按需唤醒”的能力，物理跳过闲置神经元的读取与点积计算，实现**虚室生白**。

## 核心架构与实验探索

### 1. 读写分离的非对称架构 (Asymmetric Prefill-Decode)
在完全未知的虚构知识（科幻报道）盲测中，我们验证了大模型被过度稀疏化时的“失忆症陷阱”：在阅读阶段强制休眠大量神经元，会直接摧毁模型的 In-Context Learning（上下文学习）编码能力。由此确立端侧稀疏推理的物理准则：
*   **读题（Prefill）**：严格保持 **100% 满血（Dense）** 计算，确保外部新知识与长程逻辑无损写入 KV Cache。
*   **答题（Decode）**：启动 **免训练动态稀疏 / 剪枝 / 聚类路由**，规避时序残差稀疏（$\Delta X = X_t - X_{t-1}$）在自回归闭环中的滚雪球误差爆炸，以极低带宽消耗完成文本生成。

### 2. 多路线稀疏与加速实现
围绕“保精度”与“真加速”，仓库内实现了四条互补的技术支线：
1.  **2D 分块动态绝对稀疏 + 融合 Metal 算子（`dynamic_sparsity/`）**：利用离线协方差聚类重排权重（`weight_clustering/`），按 `64x64` 分块计算免训练门控 `Impact = W_norm * X_norm`；Decode 阶段通过内置线程组并行双调排序（Bitonic Sort）的自定义 Metal Kernel 直接跳过低贡献块。
2.  **1D 结构化行列聚类稀疏（`dynamic_sparsity_1d/`）**：采用 `Balanced KMeans` 对加权激活特征离线聚类，对比评估 1D 列稀疏与行稀疏在常识推理与开放生成上的精度边界。
3.  **上下文感知物理切片剪枝（`contextual_sparsity/`）**：在 Dense Prefill 阶段实时统计当前 Prompt 的活跃神经元画像，进入 Decode 前对 MLP 权重执行物理张量切片并释放未激活统一内存，直接复用原生稠密算子提速。
4.  **粗粒度聚类专家路由 MacMoE（`clustered_moe/`）**：将稠密 MLP 离线切分为均衡专家子模块，基于专家权重质心构建免训练序列级 Top-K 路由。

## 环境配置与快速开始

本项目依赖 Apple Silicon (M1/M2/M3/M4) macOS 环境与 [uv](https://docs.astral.sh/uv/) 包管理器：

```bash
# 安装并同步 Python 3.12 虚拟环境依赖
uv sync

# 运行 2D 动态绝对稀疏（含融合 Metal Kernel）推理验证
uv run python dynamic_sparsity/dynamic_sparse_inference.py

# 运行上下文物理切片剪枝推理验证
uv run python contextual_sparsity/contextual_sparse_inference.py

# 运行粗粒度聚类 MacMoE 吞吐基准
uv run python clustered_moe/moe_inference.py
```

## 仓库模块导引

*   `weight_clustering/`：离线校准语料构建、激活协方差分析与权重列重排脚本。
*   `dynamic_sparsity/`：2D 分块动态绝对稀疏推理（含自定义 Metal 算子）及多维度评测套件（MT-Bench / HellaSwag / WikiText-2 PPL）。
*   `dynamic_sparsity_1d/`：基于 Balanced KMeans 的 1D 结构化列/行动态稀疏实现与评测报告。
*   `contextual_sparsity/`：Prefill 激活画像采集器与 Decode 物理切片剪枝推理管线。
*   `clustered_moe/`：免训练神经元聚类专家切分与序列级 `MacMoE` 路由实现。
*   `docs/`：核心架构沉淀文档（非对称稀疏原理、任务自适应稀疏曲线规律、平台化部署分析）。
*   `utils/`：支持 MLX / PyTorch 与信号中断安全落盘的通用 `Checkpointer` 基础设施。
