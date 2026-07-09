# Unlimited-OCR 昇腾 NPU 适配版

本仓库是 `baidu/Unlimited-OCR` 的昇腾 Ascend NPU 适配版本，主要用于在 Ascend 910 环境下通过 `Transformers + torch_npu + FastAPI` 跑通 OCR 推理，并提供 OCRBench 场景文字识别子集测试脚本和适配报告。

本仓库不是百度官方发布版本，而是工程适配与测试版本。上游许可证见 `LICENSE`。

## 仓库内容

- `serve_npu.py`：NPU 推理 FastAPI 服务。
- `benchmarks/run_ocrbench_str.py`：OCRBench 经典场景文字识别子集基础参数测试脚本。
- `benchmarks/run_ocrbench_str_gpu_params.py`：使用 GPU 侧参数对齐的 NPU 测试脚本。
- `docs/adaptation_report.md`：NPU 部署适配报告。
- `docs/gpu_npu_adaptation_report.md`：GPU/NPU 对齐测试与代码修改说明。
- `docs/ocrbench_str_report.md`：OCRBench STR 子集测试报告。
- `docs/sota_comparison.md`：SOTA 口径对比说明。
- `examples/`：已完成测试的 summary 示例结果。

以下大文件不会上传到 GitHub：

- 模型权重，例如 `model-00001-of-000001.safetensors`
- OCRBench parquet 数据文件
- 导出的测试图片
- 服务日志、缓存、运行输出目录

## 测试结论

在 Ascend 910 上，`Unlimited-OCR` 可以通过以下路线完成推理：

```text
Transformers + torch_npu + FastAPI
```

当前测试环境下，`vLLM-Ascend` 不能直接 serve 该模型，核心原因是 vLLM-Ascend 尚不支持模型架构：

```text
UnlimitedOCRForCausalLM
```

OCRBench 经典场景文字识别子集测试结果：

```text
49 / 76 = 64.47%
```

GPU 与 NPU 在同一批 76 张样本上的 ACC 一致。补充使用 GPU 侧推理参数在 NPU 上重跑后，结果仍为：

```text
49 / 76 = 64.47%
```

并且两轮 NPU 测试的逐样本归一化预测文本完全一致。

## 模型文件准备

请从官方模型页下载权重：

```text
https://huggingface.co/baidu/Unlimited-OCR
```

将权重放到模型目录中，例如：

```text
model-00001-of-000001.safetensors
model.safetensors.index.json
```

其中 `.safetensors` 大文件已被 `.gitignore` 排除，不会进入仓库。

## 关于官方 `wheel/` 目录

百度官方仓库中有一个 `wheel/` 目录，里面放的是 SGLang 路线使用的本地安装包，例如：

```text
wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
```

这个文件只在走官方 SGLang 推理路线时需要。本适配仓库当前使用的是：

```text
Transformers + torch_npu + FastAPI
```

因此，已验证的昇腾 NPU 服务不依赖这个 SGLang wheel。

为了让仓库保持轻量，`wheel/` 被有意排除在本仓库之外。如果后续需要尝试官方 SGLang 路线，可以从百度官方仓库下载：

```text
https://github.com/baidu/Unlimited-OCR/tree/main/wheel
```

## 启动 NPU 服务

示例启动命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=8 \
UNLIMITED_OCR_MODEL=/mnt/model/baidu/Unlimited-OCR \
uvicorn serve_npu:app \
  --host 0.0.0.0 \
  --port 10080 \
  --app-dir /mnt/model/baidu/Unlimited-OCR
```

健康检查：

```bash
curl -s http://127.0.0.1:10080/health
```

示例 OCR 请求：

```bash
curl -s -X POST http://127.0.0.1:10080/ocr \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/path/to/image.png",
    "prompt": "<image>document parsing.",
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": true,
    "max_length": 4096,
    "no_repeat_ngram_size": 35,
    "ngram_window": 128
  }'
```

接口支持两种图片输入：

- `image_path`：本机图片路径，适合 benchmark 和内网服务。
- `image_base64`：base64 图片，适合外部系统调用。

## OCRBench 测试

OCRBench 数据文件：

```text
https://huggingface.co/datasets/echo840/OCRBench/resolve/main/data/test-00000-of-00001.parquet
```

脚本默认读取路径：

```text
/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/test-00000-of-00001.parquet
```

基础参数测试：

```bash
python3 benchmarks/run_ocrbench_str.py
```

GPU 参数对齐测试：

```bash
python3 benchmarks/run_ocrbench_str_gpu_params.py
```

GPU 参数对齐测试使用：

```text
prompt=<image>document parsing.
base_size=1024
image_size=640
crop_mode=True
max_length=4096
no_repeat_ngram_size=35
ngram_window=128
```

## 重要说明

本测试更适合作为部署可用性和 GPU/NPU 适配一致性的 sanity check。`Unlimited-OCR` 的主要能力方向是长文档、整页 OCR 和结构化解析，不是单词级 cropped scene text recognition。因此 OCRBench STR 子集结果不能直接代表模型在文档解析场景下的最终能力，也不应直接与专用 STR 模型的 SOTA 分数横向比较。

如果要评估真实业务效果，建议继续测试：

- 实际业务 PDF
- 扫描件
- 表格/票据
- 多页文档
- 中文文档
- Markdown 结构化输出质量
