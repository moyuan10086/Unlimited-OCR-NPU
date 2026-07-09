# Unlimited-OCR 与 OCR SOTA 对比备注

时间：2026-07-03

## 1. 结论先行

这次本机测试的 `64.47%`，不能直接理解为 `Unlimited-OCR` 官方能力很弱。更准确的解释是：

- 本机测的是 OCRBench 里抽出来的经典场景文字识别子集，偏“裁剪单词识别”。
- `Unlimited-OCR` 的主要优势是长文档/整页 OCR 到 Markdown/结构化解析，官方主打的是 OmniDocBench 这类文档解析 benchmark。
- 在传统 STR 场景里，专用识别模型的 SOTA 已经非常高，很多经典集接近 98% 到 99%。
- 所以这次结果更像“部署后的功能 sanity check”，不是它官方主战场的 SOTA 对齐测试。

## 2. 本机结果回顾

测试集：OCRBench Hugging Face 镜像中的经典 STR 子集。

| 子集 | 样本数 | 本机 Exact Accuracy |
|---|---:|---:|
| svt | 17 | 76.47% |
| IC13_857 | 18 | 72.22% |
| IIIT5K | 15 | 66.67% |
| ct80 | 9 | 66.67% |
| IC15_1811 | 8 | 50.00% |
| svtp | 9 | 33.33% |
| 合计 | 76 | 64.47% |

明细：

- `/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/results.csv`
- `/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/summary.json`

## 3. 经典 STR SOTA 参考

经典场景文字识别任务一般看 cropped word image 的 word accuracy，常见数据集包括 IIIT5K、SVT、IC13、IC15、SVTP、CUTE80。

从公开 SOTA 聚合页看，当前经典 STR 任务已经比较饱和：

| 数据集/口径 | 公开 SOTA 参考 | 分数 |
|---|---|---:|
| IIIT5K test | DTrOCR / Cheng et al. / CA-FCN 等不同榜单项 | 99.6% - 99.8% |
| SVT test | CLIP4STR-H / Shi et al. 等 | 99.1% - 99.2% |
| IC13 test | CLIP4STR-L / ABINet 等 | 97.8% - 99.42% |
| IC15 test | DTrOCR / CSD-D 等 | 92.7% - 93.5% |
| SVTP test | DTrOCR / CSD-B / PARSeq-H 等 | 97.67% - 98.6% |
| CUTE80 test | CLIP4STR / PARSeq / CSD-D 等 | 99.65% - 99.7% |
| 6 个经典集平均 | CLIP4STR-L* | 97.42% |

参考来源：

- https://www.wizwand.com/task/scene-text-recognition
- https://www.codesota.com/browse/computer-vision/scene-text-recognition

判断：

- 如果目标是“单词级英文场景文字识别”，`Unlimited-OCR` 不是最合适的 SOTA 路线。
- 专用 STR 模型如 PARSeq、CLIP4STR、DTrOCR、ABINet 系列更贴任务，经典集分数明显更高。

## 4. OCRBench v2 通用多模态 OCR SOTA

OCRBench v2 是更偏大多模态模型的 OCR 能力评测，包含 recognition、referring、spotting、extraction、parsing、calculation、understanding、reasoning 等子任务。它比传统 STR 更综合，也更难。

官方 OCRBench v2 2026.06 English private leaderboard 中：

| 口径 | 当前靠前模型 | 分数 |
|---|---|---:|
| English Average | KDL Frontier | 68.1 |
| English Average，开源权重 | NVIDIA Nemotron 3 Nano Omni 30B A3B | 65.8 |
| English Recognition 单项 | TeleMM-2.0 | 76.4 |
| English Recognition，开源权重 | GLM-4.6V-Flash / Qwen3.6-35B-A3B 等 | 75.3 / 74.9 |

官方 OCRBench v2 2026.06 Chinese private leaderboard 中：

| 口径 | 当前靠前模型 | 分数 |
|---|---|---:|
| Chinese Average | TeleMM-2.0 | 66.2 |
| Chinese Average，开源权重 | Qwen3.5-9B | 64.1 |
| Chinese Recognition 单项 | Qwen3.5-35B-A3B | 72.6 |

参考来源：

- https://99franklin.github.io/ocrbench_v2/
- https://arxiv.org/abs/2501.00321
- https://github.com/Yuliang-Liu/MultimodalOCR

判断：

- 本机 `64.47%` 是 OCRBench v1 经典 STR 抽样子集的 exact accuracy，不能和 OCRBench v2 的私有集 average 或 recognition 单项直接横向比较。
- 但从数量级看，本机结果接近“通用 VLM 在复杂 OCR 任务上的 recognition 分数带”，明显低于专用 STR 模型在经典裁剪单词集上的 97%+。

## 5. Unlimited-OCR 官方主战场：OmniDocBench

`Unlimited-OCR` 论文和模型卡主打长文档、整页 PDF/document parsing。官方论文中，`Unlimited-OCR` 在 OmniDocBench 上的结果：

| Benchmark | Unlimited-OCR | 备注 |
|---|---:|---|
| OmniDocBench v1.5 | 93.23 overall | 论文表格中高于 DeepSeek-OCR / DeepSeek-OCR 2 |
| OmniDocBench v1.6 | 93.92 overall | 论文称 end-to-end SOTA |

论文还说明：

- 模型总参数约 3B，激活约 0.5B。
- 使用 Reference Sliding Window Attention，目标是让长文档推理时 KV cache 保持近似常量。
- 在 OmniDocBench 中报告 5580 tokens/s/512 concurrency，相比 DeepSeek-OCR 4951 tokens/s 有约 12.7% 提升。

参考来源：

- https://arxiv.org/abs/2606.23050
- https://huggingface.co/baidu/Unlimited-OCR

第三方榜单 CodeSOTA 也收录了 OCR/document 方向分数，但其页面明确提示部分顶部结果是 vendor self-reported，建议最终仍跑私有评测：

- https://www.codesota.com/ocr

## 6. 对本机适配测试的影响判断

部署层面：

- 本机 NPU 服务能跑通，说明 Ascend + Transformers + `torch_npu` shim 路线可用。
- `vllm-ascend` 当前不能直接 serve 这个架构，属于推理框架适配缺口，不代表模型本身不能在 NPU 上推理。

效果层面：

- 本机 STR 抽样准确率 `64.47%` 低于专用 STR SOTA，差距明显。
- 这个差距主要来自任务错位：`Unlimited-OCR` 面向长文档/整页解析，不是专门为 cropped word recognition 排 SOTA。
- 另外本次 prompt 使用 `<image>document parsing.`，可能促使模型输出检测标签和版面解析结构；若只测单词识别，应进一步尝试更短、更直接的文本识别 prompt，并加入输出清洗。
- Ascend 适配本身可能影响性能速度，但从现象看，不应把识别精度差距主要归因于 NPU。更大的因素是评测口径、prompt 和任务类型。

## 7. 建议下一步

如果要判断 `Unlimited-OCR` 是否真的达到官方能力，应改跑更贴近它主战场的数据：

- OmniDocBench 或类似整页文档解析集。
- PDF/扫描件到 Markdown 的结构化输出评估。
- 表格、公式、阅读顺序、段落文本 edit distance。

如果业务目标是单行/单词级 OCR：

- 对比 PaddleOCR、PARSeq、CLIP4STR、DTrOCR、ABINet 等专用 STR/OCR pipeline。
- 使用完整 IIIT5K/SVT/IC13/IC15/SVTP/CUTE80 测试集，而不是 OCRBench 抽样子集。

如果业务目标是文档 OCR：

- 继续保留 `Unlimited-OCR`，但用整页文档集重新评估。
- 优先测试实际业务 PDF、合同、报告、扫描件、表格、票据等数据。
- 对输出 markdown/文本做 edit distance、表格 TEDS、阅读顺序指标，而不是只看单词 exact match。
