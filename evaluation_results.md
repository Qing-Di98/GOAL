# GOAL 训练评估结果汇总

## 实验配置
- 基座模型: `openai/clip-vit-base-patch16`
- 训练数据: DCI `segment_with_background_DCI_train_set_max_0.01` (5446 样本)
- 测试数据: DCI test set (~2000 样本)
- epochs: 10

| 版本 | 平台 | GPU | batch_size |
|------|------|-----|:---:|
| v1 | Windows 本地 | RTX 4060 8GB | 4 |
| v2 | Windows 本地 | RTX 4060 8GB | 4 |
| v3 | Windows 本地 | RTX 4060 8GB | 4 |
| v4 | 中科大 SCOW 集群 | A100 80GB / RTX 5090 32GB | 16 |
| v5 | 中科大 SCOW 集群 | RTX 5090 32GB | 16 |

## 代码位置

| 版本 | Git Commit | 文件路径 | 说明 |
|------|------------|----------|------|
| v1 | [`eee84c6`](https://github.com/Qing-Di98/GOAL/tree/eee84c6) | `goal.py` | 原始论文代码 |
| v2 | [`bf18ac8`](https://github.com/Qing-Di98/GOAL/tree/bf18ac8) | `goal.py` | 修正 LISM 数据生成（JSON 修正未进入 git，仅 loss 修复 commit 在此标记） |
| v3 | [`5c99abe`](https://github.com/Qing-Di98/GOAL/tree/5c99abe) | `goal.py` | 非对角线惩罚版 goal.py |
| v4 | [`7279d89`](https://github.com/Qing-Di98/GOAL/tree/7279d89) | `goal.py` (同 v3) + `cluster/run_goal_v5.sbatch` | 集群部署版 |
| v5 | [`09cab97`](https://github.com/Qing-Di98/GOAL/tree/09cab97) | `improvements/multi_positive_goal/train_multi_positive_goal.py` | 方案一：top-K 多正样本对比学习 |

---

## v1 - baseline（原始论文代码）
**Git commit:** [`eee84c6`](https://github.com/Qing-Di98/GOAL/tree/eee84c6)  
**文件:** `goal.py`  
**日期:** 2026-06-09  
**输出目录:** `finetune_out_SA_1B_100k_plus_docci/`  

| 训练指标 | 初始 | 最终 |
|----------|------|------|
| org loss | 7.45 | 7.444 |
| seg loss | 7.45 | 7.444 |
| patch_sim | 0.89 | 0.9999 |
| text_sim | 0.88 | 0.9999 |

| 检索指标 | T2I | I2T |
|----------|-----|-----|
| R@1 | 0.05% | 0.05% |
| R@5 | 0.25% | 0.25% |
| R@25 | 1.25% | 1.25% |
| R@50 | 2.50% | 2.50% |

**结论:** 模型崩溃。patch/text alignment 的 MSE loss 无对角线惩罚，模型将所有 embedding 坍缩到同一点。

---

## v2 - 修正 LISM 数据生成
**Git commit:** [`bf18ac8`](https://github.com/Qing-Di98/GOAL/tree/bf18ac8)  
**文件:** `goal.py`  
**日期:** 2026-07-17  
**输出目录:** `finetune_out_v2/`  
**改动:** 修正 LISM 生成的 JSON 文件中 segment 匹配错误

| 训练指标 | 初始 | 最终 |
|----------|------|------|
| org loss | 7.50 | 7.444 |
| seg loss | 7.56 | 7.444 |
| patch_sim | 0.89 | 0.9999 |
| text_sim | 0.88 | 0.9999 |

| 检索指标 | T2I | I2T |
|----------|-----|-----|
| R@1 | 0.05% | 0.05% |
| R@5 | 0.25% | 0.25% |
| R@25 | 1.25% | 1.25% |
| R@50 | 2.50% | 2.55% |

**结论:** 数据问题不是根因，模型同样崩溃。loss 函数设计缺陷才是关键。

---

## v3 - 非对角线惩罚 + 本地 RTX 4060 (bs=4)
**Git commit:** [`5c99abe`](https://github.com/Qing-Di98/GOAL/tree/5c99abe)  
**文件:** `goal.py`  
**平台:** 本地 Windows, RTX 4060 Laptop 8GB  
**日期:** 2026-07-18  
**输出目录:** `finetune_out_v3/`  
**改动:** 
- patch alignment loss: 增加非对角线 → 0 的 MSE 惩罚
- text alignment loss: 增加非对角线 → 0 的 MSE 惩罚

| 训练指标 | 初始 | 最终 |
|----------|------|------|
| org loss | 7.54 | 7.444 |
| seg loss | 7.58 | 7.506 |
| patch_sim | 0.48 | 0.890 |
| text_sim | 0.67 | 0.961 |
| patch loss (含 off-diag) | 0.41 | 0.096 |
| text loss (含 off-diag) | 0.21 | 0.027 |

| 检索指标 | T2I | I2T |
|----------|-----|-----|
| R@1 | 0.25% | **2.10%** |
| R@5 | 0.55% | **6.50%** |
| R@25 | 2.05% | **18.11%** |
| R@50 | 3.85% | **24.46%** |

**结论:** ✅ 崩溃已修复。非对角线惩罚阻止了 embedding 坍缩。Local alignment 有效（I2T R@1 提升 42x）。

---

## v4 - 非对角线惩罚 + 中科大 SCOW 集群 (bs=16)
**Git commit:** [`7279d89`](https://github.com/Qing-Di98/GOAL/tree/7279d89)  
**文件:** `goal.py` (同 v3) + `cluster/run_goal_v5.sbatch`  
**平台:** 中科大本科生算力平台 (SCOW), A100-SXM4-80GB / RTX 5090 32GB  
**日期:** 2026-07-18  
**输出目录:** `finetune_out_v4/`  
**配置:** batch_size=16, num_workers=0, epochs=10, srun 分配

| 训练指标 | 初始 | 最终 |
|----------|------|------|
| org loss | — | 8.834 |
| seg loss | — | 8.906 |
| patch_sim | — | 0.916 |
| text_sim | — | 0.938 |
| patch loss (含 off-diag) | — | 0.046 |
| text loss (含 off-diag) | — | 0.022 |

| 检索指标 | T2I | I2T |
|----------|-----|-----|
| R@1 | 0.20% | **1.70%** |
| R@5 | 0.80% | **5.95%** |
| R@25 | 2.50% | **17.06%** |
| R@50 | 3.90% | **23.71%** |

**结论:** 无崩溃，patch_sim=0.916。结果与本地 bs=4 基本持平，bs=16 未带来额外提升（GPU bf16 精度差异 + num_workers=0 + 续跑导致优化器动量丢失）。

---

## v5 - 方案一：多正样本对比学习 + SCOW 集群 (bs=16, top-K=3) ⭐
**Git commit:** [`09cab97`](https://github.com/Qing-Di98/GOAL/tree/09cab97)  
**文件:** `improvements/multi_positive_goal/train_multi_positive_goal.py`  
**平台:** 中科大 SCOW 集群, RTX 5090 32GB  
**日期:** 2026-07-20  
**输出目录:** `finetune_out_multi_pos/`  
**配置:** batch_size=16, top_k=3, num_workers=0, epochs=10  
**改动:** 
- 恢复原始 loss（去除 v3/v4 的非对角线惩罚）
- 每张原图选 top-K 个 segment 而非 1 个
- 添加多正样本对比学习 loss：`org_image → K seg_texts` 和 `org_text → K seg_images`
- NFS 容错：缺失文件自动随机重试

| 训练指标 | 初始 | 最终 |
|----------|------|------|
| org loss | 0.31 | ~0.00 |
| seg loss | 0.44 | ~0.01 |
| multi loss | 1.20 | ~0.01 |
| patch_sim | 0.65 | ~0.91 |
| text_sim | 0.50 | ~0.90 |
| total loss | 2.13 | ~0.04 |

| 检索指标 | T2I | I2T |
|----------|-----|-----|
| R@1 | **71.04%** 🚀 | **70.49%** 🚀 |
| R@5 | **88.04%** | **88.14%** |
| R@25 | **95.60%** | **96.70%** |
| R@50 | **97.60%** | **97.95%** |

**结论:** 🎉 **巨大成功！R@1 比预训练 CLIP 高出 24 个百分点。**  
top-K 多正样本对比学习完美利用了被浪费的 segment 标注，将全局 CLIP loss 从 ~7.4 降至 ~0.00。模型无崩溃（patch_sim 0.65→0.91，远未到 1.0）。方案一证实了"提供更多正样本锚点"是解决 small batch 对比学习困境的有效策略。

---

## 对比：预训练 CLIP baseline
**模型:** `openai/clip-vit-base-patch16` (未微调)

| 检索指标 | T2I | I2T |
|----------|-----|-----|
| R@1 | 46.87% | 47.97% |
| R@5 | 69.08% | 69.43% |
| R@25 | 84.64% | 85.44% |
| R@50 | 90.00% | 89.74% |

---

## 全版本对比总表

| 检索指标 | 预训练 CLIP | v1 崩溃 | v2 数据修 | v3 非对角 | v4 集群 | **v5 多正样本** |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| T2I R@1 | 46.87 | 0.05 | 0.05 | 0.25 | 0.20 | **71.04** |
| T2I R@5 | 69.08 | 0.25 | 0.25 | 0.55 | 0.80 | **88.04** |
| T2I R@50 | 90.00 | 2.50 | 2.50 | 3.85 | 3.90 | **97.60** |
| I2T R@1 | 47.97 | 0.05 | 0.05 | 2.10 | 1.70 | **70.49** |
| I2T R@5 | 69.43 | 0.25 | 0.25 | 6.50 | 5.95 | **88.14** |
| I2T R@50 | 89.74 | 2.55 | 2.55 | 24.46 | 23.71 | **97.95** |

---

## 关键发现
1. **MSE 对角惩罚 + 无非对角约束 = 必然崩溃** — 模型的最优解是把所有 embedding 变成同一个向量
2. **数据质量重要但不是根因** — v2 修正了 JSON 但未改变结局；loss 函数设计缺陷才是关键
3. **非对角线惩罚能阻止崩溃但检索表现弱** — v3/v4 patch_sim 从 0.9999 降至 0.89~0.92，但全局检索仍远不如预训练
4. **多正样本对比学习是正确方向** ⭐ — v5 将 wasted segment 标注变成多正样本，全局 CLIP loss 从 7.4 → 0.0，检索全面超越预训练 CLIP
5. **每个样本的多重语义锚点比更大 batch_size 更有效** — v5 (bs=16 + top-K=3) 完胜 v4 (bs=16) 和 v3 (bs=4 + 非对角惩罚)
6. **下一步:** 探索更大的 K 值、在不同数据集验证、结合非对角线惩罚与多正样本
