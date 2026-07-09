# Unlimited-OCR 经典 OCR 识别数据集测试报告

测试时间：2026-07-03  
测试对象：`baidu/Unlimited-OCR`  
运行方式：Transformers + `torch_npu` FastAPI 服务  
服务地址：`http://127.0.0.1:10080`

## 1. 测试目的

本次测试的目标是从公开数据集中选取经典 OCR 场景文字识别样本，对当前 Ascend NPU 适配后的 `Unlimited-OCR` 服务做一次功能与基础效果验证。

由于 `vllm-ascend` 当前不能直接加载该模型，报错为 `UnlimitedOCRForCausalLM architecture not supported`，因此本轮测试使用已经跑通的 Transformers + `torch_npu` 服务，而不是 vLLM serve。

## 2. 数据集来源

本次选择 OCRBench 的 Hugging Face 镜像数据：

- 数据集页：https://huggingface.co/datasets/echo840/OCRBench
- OCRBench 论文：https://arxiv.org/abs/2305.07895
- OCRBench 官方代码仓库：https://github.com/Yuliang-Liu/MultimodalOCR

选择原因：

- OCRBench 是面向多模态模型 OCR 能力的综合评测集，论文中说明其覆盖 Text Recognition、Scene Text-Centric VQA、Document-Oriented VQA、KIE、HMER 等任务。
- 其中 Text Recognition 部分包含多个经典场景文字识别数据集子集，适合快速验证模型对单词级/短文本识别的能力。
- Hugging Face 上可直接下载 parquet，便于在当前服务器上快速复现。

本地文件：

- 下载文件：`/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/test-00000-of-00001.parquet`
- 评测脚本：`/mnt/model/baidu/Unlimited-OCR/benchmarks/run_ocrbench_str.py`
- 输出目录：`/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval`

## 3. 选取的经典识别子集

本轮只抽取 OCRBench 中的经典场景文字识别子集：

| 子集 | 类型 | 样本数 |
|---|---:|---:|
| IIIT5K | Regular Text Recognition | 15 |
| svt | Regular Text Recognition | 17 |
| IC13_857 | Regular Text Recognition | 18 |
| IC15_1811 | Irregular Text Recognition | 8 |
| svtp | Irregular Text Recognition | 9 |
| ct80 | Irregular Text Recognition | 9 |
| 合计 | - | 76 |

说明：这里使用的是 OCRBench 汇总评测里的抽样子集，不是 IIIT5K、SVT、IC13、IC15、SVTP、CUTE80/CT80 等原始数据集的完整测试集，因此结果不能作为这些原始数据集的官方完整精度。

## 4. 测试方法

服务调用：

- 接口：`POST /ocr`
- 图片来源：从 parquet 中提取 image bytes，保存为 png 后传入服务
- Prompt：`<image>document parsing.`
- `max_length`：1024
- 超时：120 秒

评估指标：

- `exact_acc`：预测文本归一化后与答案完全一致
- `contains_acc`：预测文本归一化后包含答案

归一化规则：

- 去除模型输出中的检测标签，例如 `<|det|>...<|/det|>`
- 去除其他尖括号标签
- 转小写
- 删除非英文字母和数字字符

本轮 `exact_acc` 与 `contains_acc` 相同，说明正确样本基本都是精确匹配，错误样本也没有出现“答案被包含在更长预测文本中”的情况。

## 5. 总体结果

| 指标 | 结果 |
|---|---:|
| 样本数 | 76 |
| Exact 正确数 | 49 |
| Exact Accuracy | 64.47% |
| Contains 正确数 | 49 |
| Contains Accuracy | 64.47% |
| 平均耗时 | 0.732 秒/张 |
| 总耗时 | 56.457 秒 |

服务健康状态：

```json
{"status":"ok","npu_free":55720800256,"npu_total":65787658240}
```

## 6. 分数据集结果

| 子集 | 样本数 | Exact | Exact Accuracy | 平均耗时 |
|---|---:|---:|---:|---:|
| svt | 17 | 13 | 76.47% | 0.743s |
| IC13_857 | 18 | 13 | 72.22% | 0.730s |
| IIIT5K | 15 | 10 | 66.67% | 0.725s |
| ct80 | 9 | 6 | 66.67% | 0.753s |
| IC15_1811 | 8 | 4 | 50.00% | 0.700s |
| svtp | 9 | 3 | 33.33% | 0.735s |

观察：

- 规则文字识别子集表现更稳定，`svt` 和 `IC13_857` 分别达到 76.47% 和 72.22%。
- 不规则文字识别子集中，`svtp` 最弱，只有 33.33%；`IC15_1811` 为 50.00%。
- 模型输出经常包含定位标记 `<|det|>...<|/det|>`，需要后处理剥离后再评估纯文本。
- 部分失败样本是空文本，模型只输出了检测区域；部分失败样本是字符级误识别，例如漏字、相近字符混淆。

## 7. 样例

正确样例：

| 子集 | 标注 | 模型输出归一化后 |
|---|---|---|
| IIIT5K | FRIEND | friend |
| IIIT5K | CHAIN | chain |
| IIIT5K | MARKET | market |
| ct80 | hutchinson | hutchinson |
| ct80 | mobile | mobile |

错误样例：

| 子集 | 标注 | 模型输出归一化后 | 现象 |
|---|---|---|---|
| IIIT5K | CENTRE | 空 | 只输出检测框 |
| IIIT5K | EXTRA | extr | 字符缺失/重音字符影响 |
| IIIT5K | COTTAGE | cotage | 漏字 |
| svtp | METHODIST | hethodist | 首字符误识别 |
| ct80 | wigan | nigan | 首字符误识别 |

完整明细见：

- `/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/results.csv`
- `/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/summary.json`

## 8. 复现命令

确认服务健康：

```bash
curl -s http://127.0.0.1:10080/health
```

运行评测：

```bash
python3 /mnt/model/baidu/Unlimited-OCR/benchmarks/run_ocrbench_str.py
```

查看汇总：

```bash
cat /mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/summary.json
```

## 9. 结论

当前 Ascend NPU 上的 Transformers + `torch_npu` 服务可以完成 OCRBench 经典场景文字识别子集推理，服务稳定，平均单图耗时约 0.73 秒。

效果上，规则场景文字识别可用性较好；不规则、透视、弯曲或低质量文字识别仍有明显误识别。该结果更适合作为“部署可用性 + 基础识别效果”的 sanity check，不建议直接作为正式 OCR 精度结论或官方榜单分数。

后续如果要做更严谨评测，建议：

- 跑完整 OCRBench 1000 条样本，而不是只跑经典 STR 子集。
- 使用官方 OCRBench 评测脚本对齐打分口径。
- 针对 `Unlimited-OCR` 调整 prompt，分别测试 document parsing、纯文本识别、检测+识别等不同模式。
- 增加中文、文档、票据、手写、公式等任务集，覆盖模型实际业务输入。
