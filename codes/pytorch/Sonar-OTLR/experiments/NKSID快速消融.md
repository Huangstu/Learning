# NKSID 快速消融

2026-11-25，使用 GTX 1650、现有 Conda py311、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124。

从项目根目录运行：

```powershell
git clone --depth 1 https://github.com/Jorwnpay/NK-Sonar-Image-Dataset.git external/NKSID
python scripts/prepare_nksid.py
python scripts/train_satml.py --config configs/nksid_quick.json
python scripts/train_satml.py --evaluate checkpoints/nksid_quick/satml.pt
```

仓库已下载时可跳过 clone。配置需要 checkpoints/pretrained/resnet18-f37072fd.pth，项目中已下载。权重来源 https://download.pytorch.org/models/resnet18-f37072fd.pth，SHA-256 为 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec。

- 数据：官方 NKSID 提交 e9a9e9843f0465f3e196f715d5650120abd99957，解包 2,617 张。按解码像素做精确去重。训练 524、已知验证 124、测试 259（已知 124、未知 135）。未知类是 iron_pipeline、small_propeller、soft_pipeline。训练类中 fishing_net 有 12 张，其余各 128 张。图像级划分无法排除跨集合的近重复帧或同目标实例。
- 源码：ResNet-18 空间特征、CAM/Gini/分位数软掩码及无自配对混合；尾类 FIFO、EMA 原型、历史相似度分位筛选、按类别平均的监督对比。开放损失参考论文 §5.1 及官方 PLUD 代码的 sigmoid logit 推拉（external/PLUD，提交 68bd8573cb23a2acb828b502ad4d7409afd80804）。
- 协议：统一 ImageNet 初始化、固定种子 42、96×96、AdamW、5 轮。各组分类损失均使用批次后半部，前半部在启用 SAPFS 时用于合成。基线为交叉熵加验证阈值，SAPFS 组同时新增合成和开放损失，此对比无法单独隔离两者效应。
- 快速设置：尾类频数下限设为 16，避免论文的 300 将截断后的全部已知类视为尾类。对比温度 0.07、相似度缓冲长度 32、掩码停止梯度、配对采用随机循环及其他预处理和优化器设置为本实现选择。并非官方实现或论文五组五折复现。
- 指标：已知 Top-1 忽略拒识。已知正确且接受率与未知召回按已知验证集约 95% 接受率确定的阈值计算。OSCR_macro 对各已知类正确识别且接受率做宏平均，并对未知误接受率积分。

| 模型 | 已知 Top-1 | 已知正确且接受 | 未知拒绝召回 | OSCR_macro |
|---|---:|---:|---:|---:|
| 基线 | 95.16% | 90.32% | 65.93% | 85.49% |
| SAPFS | 96.77% | 95.16% | 45.93% | 87.20% |
| PTMCL | 96.77% | 95.97% | 42.22% | 81.89% |
| SATML | 97.58% | 95.97% | 33.33% | 76.22% |

本次完整模型提升已知 Top-1，但开放集表现下降。SAPFS 单独使用的 OSCR_macro 最高。单次划分不足以判断稳定性。两组启用 PTMCL 的运行均有 105 个有效对比更新步，并累计选用 364 个历史特征（含重复使用）。完整模型权重重新加载后指标一致。

逐样本分数、固定划分和完整指标在 outputs/nksid_quick，四组权重在 checkpoints/nksid_quick。重新训练会更新同名结果，可在配置中指定其他输出目录保留历史。
