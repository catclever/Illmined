# HellaSwag 常识推理评测报告

- 数据集: HellaSwag Validation Set (评估 100 题)
- 模型: Qwen/Qwen3-8b

| 模型版本 | 准确率 (Acc, 越高越好) | 相对损失 |
|---|---|---|
| Dense 满血版 | **55.00%** | - |
| Sparse 动态稀疏 (10%-70%) | **42.00%** | -13.00% |

## Detailed Error Analysis

- Both Correct: 38
- Both Wrong: 41
- Dense Better (Dense ✓, Sparse ✗): 17
- Sparse Better (Sparse ✓, Dense ✗): 4

### Category Breakdown
| Category | Total | Dense Acc | Sparse Acc | Dense Won | Sparse Won |
|---|---|---|---|---|---|
| Sharpening knives | 14 | 42.9% | 35.7% | 2 | 1 |
| Playing harmonica | 6 | 33.3% | 50.0% | 0 | 1 |
| Washing face | 6 | 33.3% | 33.3% | 0 | 0 |
| Having an ice cream | 5 | 60.0% | 60.0% | 0 | 0 |
| Cutting the grass | 5 | 60.0% | 20.0% | 2 | 0 |
| Gargling mouthwash | 4 | 75.0% | 50.0% | 1 | 0 |
| Clean and jerk | 3 | 0.0% | 0.0% | 0 | 0 |
| High jump | 3 | 33.3% | 0.0% | 1 | 0 |
| Cheerleading | 3 | 33.3% | 0.0% | 1 | 0 |
| Surfing | 3 | 33.3% | 33.3% | 0 | 0 |
| Beer pong | 3 | 100.0% | 33.3% | 2 | 0 |
| Grooming dog | 3 | 100.0% | 66.7% | 1 | 0 |
| Hitting a pinata | 3 | 33.3% | 33.3% | 0 | 0 |
| Ironing clothes | 3 | 66.7% | 66.7% | 0 | 0 |
| Clipping cat claws | 3 | 66.7% | 33.3% | 1 | 0 |
| Baton twirling | 3 | 66.7% | 33.3% | 1 | 0 |
| Removing ice from car | 3 | 66.7% | 33.3% | 1 | 0 |
| Running a marathon | 2 | 50.0% | 100.0% | 0 | 1 |
| Using the pommel horse | 2 | 50.0% | 100.0% | 0 | 1 |
| Kayaking | 2 | 0.0% | 0.0% | 0 | 0 |
| Cleaning windows | 2 | 50.0% | 0.0% | 1 | 0 |
| Ice fishing | 2 | 100.0% | 50.0% | 1 | 0 |
| Roof shingle removal | 1 | 0.0% | 0.0% | 0 | 0 |
| Canoeing | 1 | 100.0% | 100.0% | 0 | 0 |
| Sumo | 1 | 0.0% | 0.0% | 0 | 0 |
| Playing bagpipes | 1 | 100.0% | 0.0% | 1 | 0 |
| Playing water polo | 1 | 100.0% | 100.0% | 0 | 0 |
| Paintball | 1 | 100.0% | 100.0% | 0 | 0 |
| Archery | 1 | 100.0% | 100.0% | 0 | 0 |
| Plastering | 1 | 100.0% | 100.0% | 0 | 0 |
| Applying sunscreen | 1 | 100.0% | 100.0% | 0 | 0 |
| Polishing shoes | 1 | 0.0% | 0.0% | 0 | 0 |
| Zumba | 1 | 100.0% | 100.0% | 0 | 0 |
| Playing violin | 1 | 0.0% | 0.0% | 0 | 0 |
| Making a sandwich | 1 | 100.0% | 0.0% | 1 | 0 |
| Making a lemonade | 1 | 100.0% | 100.0% | 0 | 0 |
| Throwing darts | 1 | 100.0% | 100.0% | 0 | 0 |
| Rollerblading | 1 | 100.0% | 100.0% | 0 | 0 |
| Layup drill in basketball | 1 | 100.0% | 100.0% | 0 | 0 |
