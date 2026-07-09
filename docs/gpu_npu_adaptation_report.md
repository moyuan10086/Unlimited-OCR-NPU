# Unlimited-OCR GPU/NPU 适配与 OCRBench 对齐测试报告

## 目录

1. 实验结论
2. 背景与目标
3. 环境对比
4. 部署路线
5. 代码修改说明
6. 目录结构
7. OCRBench 场景文字识别结果
8. GPU/NPU ACC 一致性分析
9. 已知限制
10. 复现命令
11. 最终结论

## 1. 实验结论

本次完成了 `baidu/Unlimited-OCR` 在 NVIDIA GPU 和 Ascend NPU 两种环境下的 OCRBench 场景文字识别测试。

核心结论：

- GPU Windows 环境可通过官方 README 中的 Transformers 路线完成推理。
- Ascend NPU 环境无法直接使用官方 CUDA 调用，需要做 CUDA-to-NPU 运行时适配。
- `vllm-ascend` 当前不能直接 serve 该模型，原因是模型架构 `UnlimitedOCRForCausalLM` 尚未被 vLLM-Ascend 注册支持。
- NPU 侧已通过 `Transformers + torch_npu + FastAPI` 路线跑通。
- GPU 与 NPU 在 OCRBench 经典场景文字识别 76 张样本上的 ACC 一致，均为 `49 / 76 = 64.47%`。
- 补充 GPU 参数对齐测试后，NPU 侧使用 `image_size=640`、`crop_mode=True`、`max_length=4096`、`no_repeat_ngram_size=35`、`ngram_window=128` 仍得到 `49 / 76 = 64.47%`，且与旧 NPU 结果的归一化预测文本逐样本一致。

这说明：当前 NPU 适配没有在该测试集上引入可见准确率损失，模型功能可用。但这不代表 GPU/NPU 输出字节级完全一致，也不代表该结果是 Unlimited-OCR 的官方 SOTA 能力。

## 2. 背景与目标

`Unlimited-OCR` 官方实现主要面向 CUDA/NVIDIA GPU。模型源码和推理链路中存在 `.cuda()`、`torch.autocast("cuda")` 等 CUDA 相关调用。

本次目标：

- 在 Ascend NPU 上完成模型加载和推理。
- 封装 HTTP OCR 服务，便于测试调用。
- 使用公开 OCRBench 数据集中的经典场景文字识别子集做效果验证。
- 与 GPU 侧同一批 76 张图片的 ACC 做对齐。
- 整理代码修改点和目录结构，形成可交付适配说明。

## 3. 环境对比

### 3.1 GPU 测试环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| GPU | NVIDIA Quadro RTX 6000 |
| 显存 | 24576 MiB |
| NVIDIA 驱动 | 595.71 |
| 驱动支持 CUDA | 13.2 |
| Python | 3.12.10 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| Transformers | 4.57.1 |
| Datasets | 5.0.0 |
| pandas | 3.0.3 |
| Pillow | 12.2.0 |
| PyMuPDF | 1.27.2.2 |
| Unlimited-OCR commit | 528fca4 |

GPU 侧补充说明：

- Windows 环境无法使用 SGLang 路线，因为项目自带 `sglang-kernel==0.4.1` 没有 Windows `win_amd64` wheel。
- 因此 GPU 侧采用项目 README 中的 Transformers 推理路线。

### 3.2 NPU 测试环境

| 项目 | 配置 |
|---|---|
| 操作系统 | openEuler 22.03 LTS-SP4，aarch64 |
| 加速卡 | Ascend 910 |
| 可见芯片 | 8 张 Ascend 910，共 16 个可见 chip |
| 本次使用 chip | `ASCEND_RT_VISIBLE_DEVICES=8` |
| 容器 | `unlimited-ocr-npu-test` |
| 基础镜像 | `quay.io/ascend/vllm-ascend:v0.18.0-a3` |
| Python | 3.11 |
| PyTorch | 2.9.0+cpu |
| torch_npu | 2.9.0.post1 |
| Transformers | 4.57.6 |
| vLLM | 0.18.0 |
| vLLM-Ascend | 0.18.0 |

模型文件：

| 项目 | 路径/说明 |
|---|---|
| NPU 模型目录 | `/mnt/model/baidu/Unlimited-OCR` |
| 权重文件 | `model-00001-of-000001.safetensors` |
| 权重大小 | 约 6.3 GB |
| 模型架构 | `UnlimitedOCRForCausalLM` |
| 模型类型 | `unlimited-ocr` |

## 4. 部署路线

### 4.1 GPU 路线

GPU 侧使用官方 Transformers 推理方式：

```text
Hugging Face Transformers + CUDA + 本地模型权重
```

GPU 侧测试脚本：

```text
D:\ch\Unlimited-OCR\eval_ocrbench_scene_text.py
```

GPU 侧模型路径：

```text
D:\ch\Unlimited-OCR\models\Unlimited-OCR
```

### 4.2 NPU 路线

NPU 侧采用：

```text
Transformers + torch_npu + FastAPI
```

原因：

- Transformers 可以通过 `trust_remote_code=True` 加载 `UnlimitedOCRForCausalLM`。
- `torch_npu` 可以执行模型推理。
- FastAPI 便于封装 HTTP 接口和 OCRBench 批量测试。

NPU 服务脚本：

```text
/mnt/model/baidu/Unlimited-OCR/serve_npu.py
```

NPU 服务地址：

```text
http://127.0.0.1:10080
```

服务接口：

| 接口 | 说明 |
|---|---|
| `GET /health` | 返回服务状态和 NPU HBM 信息 |
| `POST /ocr` | 输入图片路径或 base64，返回 OCR 文本和耗时 |

### 4.3 vLLM-Ascend 路线结论

尝试通过 vLLM-Ascend 原生 serve 启动失败：

```text
Model architectures ['UnlimitedOCRForCausalLM'] are not supported for now.
```

原因：

- `Unlimited-OCR` 使用自定义架构 `UnlimitedOCRForCausalLM`。
- vLLM 即使开启 `--trust-remote-code`，仍要求模型架构在 vLLM 模型注册表中受支持。
- 当前 `vllm-ascend:v0.18.0-a3` 未内置该架构。

因此，当前 NPU 可用路线是 Transformers 服务，不是 vLLM-Ascend 原生服务。

## 5. 代码修改说明

本次没有修改官方模型核心文件，例如：

- `modeling_unlimitedocr.py`
- `modeling_deepseekv2.py`
- `deepencoder.py`
- `configuration_deepseek_v2.py`

新增/修改主要集中在 NPU 服务封装和评测脚本。

### 5.1 新增 NPU 服务脚本

文件：

```text
/mnt/model/baidu/Unlimited-OCR/serve_npu.py
```

主要修改点：

1. 引入 `torch_npu`

```python
import torch_npu  # noqa: F401
```

作用：注册 NPU backend，使 PyTorch 可以识别并使用 `npu` 设备。

2. 增加 CUDA-to-NPU runtime shim

```python
def _patch_cuda_calls() -> None:
    def tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        return self.to("npu" if device is None else f"npu:{device}", non_blocking=non_blocking)

    def module_cuda(self, device=None):
        return self.to("npu" if device is None else f"npu:{device}")

    original_autocast = torch.autocast

    def autocast(device_type, *args, **kwargs):
        if device_type == "cuda":
            device_type = "npu"
        return original_autocast(device_type, *args, **kwargs)

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda
    torch.autocast = autocast
```

作用：

- 将 `tensor.cuda()` 转为 `tensor.to("npu")`。
- 将 `module.cuda()` 转为 `module.to("npu")`。
- 将 `torch.autocast("cuda")` 转为 `torch.autocast("npu")`。
- 避免直接改官方模型源码，降低后续升级冲突。

3. 使用 Transformers 加载模型到 NPU

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_safetensors=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).eval().npu()
torch.npu.synchronize()
```

作用：

- 支持远程自定义模型代码。
- 使用 `bfloat16` 降低显存/HBM 压力。
- 直接将模型移动到 NPU。

4. 封装 FastAPI 服务

```python
app = FastAPI(title="Unlimited-OCR NPU")
```

已实现接口：

- `GET /health`
- `POST /ocr`

5. 增加 NPU 显存健康检查

```python
free, total = torch.npu.mem_get_info()
return {"status": "ok", "npu_free": free, "npu_total": total}
```

作用：快速确认服务是否正常，以及当前 NPU HBM 剩余量。

6. 支持图片路径和 base64 输入

请求字段：

```python
class OCRRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    prompt: str = "<image>document parsing."
    max_length: int = 4096
    base_size: int = 1024
    image_size: int = 1024
    crop_mode: bool = False
    no_repeat_ngram_size: int = 0
    ngram_window: int = 0
```

作用：

- 本地 benchmark 使用 `image_path`。
- 外部系统调用可使用 `image_base64`。
- 可通过请求参数切换 `base_size`、`image_size`、`crop_mode`、`no_repeat_ngram_size`、`ngram_window`，用于和 GPU 侧参数严格对齐。

7. 增加线程锁

```python
lock = threading.Lock()
```

作用：

- 当前服务为单模型单 NPU 推理。
- 用锁串行化请求，避免并发请求导致 HBM 竞争或 OOM。

8. 调用模型官方 `infer`

```python
text = model.infer(
    tokenizer,
    prompt=req.prompt,
    image_file=image_path,
    output_path=OUTPUT_DIR,
    base_size=req.base_size,
    image_size=req.image_size,
    crop_mode=req.crop_mode,
    max_length=req.max_length,
    no_repeat_ngram_size=req.no_repeat_ngram_size,
    ngram_window=req.ngram_window,
    save_results=False,
    eval_mode=True,
)
```

说明：

- NPU 服务侧使用 `save_results=False`，直接返回文本。
- Benchmark 脚本负责保存图片和统计结果。
- 当前代码从请求体读取 `base_size`、`image_size`、`crop_mode`、`max_length`、`no_repeat_ngram_size`、`ngram_window`，默认值仍兼容第一轮 NPU 服务参数。

### 5.2 新增 OCRBench NPU 评测脚本

文件：

```text
/mnt/model/baidu/Unlimited-OCR/benchmarks/run_ocrbench_str.py
/mnt/model/baidu/Unlimited-OCR/benchmarks/run_ocrbench_str_gpu_params.py
```

主要功能：

- 读取 OCRBench parquet 文件。
- 筛选 6 个经典场景文字识别子集。
- 将 parquet 中的图片字节导出为 PNG。
- 调用 NPU FastAPI `/ocr` 接口。
- 对答案和预测做归一化。
- 计算 exact accuracy 和 contains accuracy。
- 输出 `results.csv` 和 `summary.json`。
- `run_ocrbench_str_gpu_params.py` 使用 GPU 侧记录的推理参数，在 NPU 10080 参数化服务上做严格对齐测试。

筛选子集：

```python
TARGETS = ["IIIT5K", "svt", "IC13_857", "IC15_1811", "svtp", "ct80"]
```

归一化规则：

```python
def norm(s):
    if s is None:
        return ""
    if isinstance(s, list):
        s = " ".join(map(str, s))
    s = str(s).strip()
    s = re.sub(r"<\|det\|>.*?<\|/det\|>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s
```

说明：

- 模型输出常带 `<|det|>...<|/det|>` 检测标签。
- 本次评测只比较纯文本识别结果，因此剥离检测标签。

### 5.3 新增报告文件

已新增/整理报告：

```text
/mnt/model/baidu/Unlimited-OCR/adaptation_report.md
/mnt/model/baidu/Unlimited-OCR/benchmarks/ocrbench_str_report.md
/mnt/model/baidu/Unlimited-OCR/benchmarks/sota_comparison.md
/mnt/model/baidu/Unlimited-OCR/gpu_npu_adaptation_report.md
```

说明：

- `adaptation_report.md`：NPU 部署适配报告。
- `ocrbench_str_report.md`：OCRBench 经典 STR 子集测试报告。
- `sota_comparison.md`：SOTA 口径对比说明。
- `gpu_npu_adaptation_report.md`：本报告，汇总 GPU/NPU 适配、ACC 对齐、代码修改和目录。

## 6. 目录结构

### 6.1 NPU 侧目录

```text
/mnt/model/baidu/Unlimited-OCR
├── README.md
├── config.json
├── configuration_deepseek_v2.py
├── conversation.py
├── deepencoder.py
├── modeling_deepseekv2.py
├── modeling_unlimitedocr.py
├── model-00001-of-000001.safetensors
├── model.safetensors.index.json
├── processor_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── serve_npu.py
├── server.log
├── test_input.png
├── adaptation_report.md
├── gpu_npu_adaptation_report.md
├── wheel
│   └── sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl
└── benchmarks
    ├── run_ocrbench_str.py
    ├── run_ocrbench_str_gpu_params.py
    ├── ocrbench_str_report.md
    ├── sota_comparison.md
    └── OCRBench
        ├── test-00000-of-00001.parquet
        ├── str_eval
        │   ├── results.csv
        │   ├── summary.json
        │   └── images
        └── str_eval_gpu_params_npu
            ├── results.csv
            ├── summary.json
            └── images
```

### 6.2 GPU 侧目录

```text
D:\ch\Unlimited-OCR
├── eval_ocrbench_scene_text.py
├── models
│   └── Unlimited-OCR
├── datasets
│   └── OCRBench
│       └── test-00000-of-00001.parquet
└── outputs
    └── ocrbench_scene_text
        ├── summary.json
        ├── details.csv
        ├── images
        └── runs
```

## 7. OCRBench 场景文字识别结果

数据集来源：

- Hugging Face：https://huggingface.co/datasets/echo840/OCRBench
- OCRBench 论文：https://arxiv.org/abs/2305.07895
- OCRBench 仓库：https://github.com/Yuliang-Liu/MultimodalOCR

测试样本：

| OCRBench 子集 | 问题类型 | 图片数 |
|---|---|---:|
| IIIT5K | Regular Text Recognition | 15 |
| svt | Regular Text Recognition | 17 |
| IC13_857 | Regular Text Recognition | 18 |
| IC15_1811 | Irregular Text Recognition | 8 |
| svtp | Irregular Text Recognition | 9 |
| ct80 | Irregular Text Recognition | 9 |
| 合计 | - | 76 |

### 7.1 GPU 结果

| 指标 | 数值 |
|---|---:|
| 正确数 | 49 |
| 总样本数 | 76 |
| 准确率 | 64.47% |

### 7.2 NPU 结果

第一轮 NPU 测试使用基础服务参数：

```text
base_size=1024
image_size=1024
crop_mode=False
max_length=1024
no_repeat_ngram_size=0
ngram_window=0
```

| 指标 | 数值 |
|---|---:|
| 正确数 | 49 |
| 总样本数 | 76 |
| 准确率 | 64.47% |
| 平均耗时 | 0.732 秒/张 |
| 总耗时 | 56.457 秒 |

NPU 分子集结果：

| 子集 | 正确数 | 总数 | 准确率 |
|---|---:|---:|---:|
| IC13_857 | 13 | 18 | 72.22% |
| IC15_1811 | 4 | 8 | 50.00% |
| IIIT5K | 10 | 15 | 66.67% |
| ct80 | 6 | 9 | 66.67% |
| svt | 13 | 17 | 76.47% |
| svtp | 3 | 9 | 33.33% |

NPU 输出文件：

```text
/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/results.csv
/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/summary.json
/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/images
```

### 7.3 NPU GPU 参数对齐结果

第二轮 NPU 测试使用 GPU 侧记录的关键推理参数：

```text
prompt=<image>document parsing.
base_size=1024
image_size=640
crop_mode=True
max_length=4096
no_repeat_ngram_size=35
ngram_window=128
```

结果：

| 指标 | 数值 |
|---|---:|
| 正确数 | 49 |
| 总样本数 | 76 |
| 准确率 | 64.47% |
| 平均耗时 | 0.805 秒/张 |
| 总耗时 | 62.024 秒 |

分子集结果：

| 子集 | 正确数 | 总数 | 准确率 | 平均耗时 |
|---|---:|---:|---:|---:|
| IC13_857 | 13 | 18 | 72.22% | 0.813s |
| IC15_1811 | 4 | 8 | 50.00% | 0.733s |
| IIIT5K | 10 | 15 | 66.67% | 0.896s |
| ct80 | 6 | 9 | 66.67% | 0.791s |
| svt | 13 | 17 | 76.47% | 0.785s |
| svtp | 3 | 9 | 33.33% | 0.755s |

输出文件：

```text
/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval_gpu_params_npu/results.csv
/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval_gpu_params_npu/summary.json
/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval_gpu_params_npu/images
```

与第一轮 NPU 结果对比：

| 对比项 | 结果 |
|---|---:|
| 样本数差异 | 0 |
| `exact` 状态差异 | 0 |
| `contains` 状态差异 | 0 |
| 归一化预测文本差异 | 0 |
| 第一轮平均耗时 | 0.732 秒/张 |
| GPU 参数对齐平均耗时 | 0.805 秒/张 |

## 8. GPU/NPU ACC 一致性分析

GPU 与 NPU 均得到：

```text
49 / 76 = 64.47%
```

说明：

- NPU 运行时 shim 没有导致该测试集上的准确率下降。
- Ascend NPU 上的 `Transformers + torch_npu` 推理链路在功能层面可用。
- NPU 适配主要解决的是设备调用兼容问题，而不是改变模型结构或权重。
- 该结果可以作为 NPU 适配正确性的初步证据。
- 补充 GPU 参数对齐后，NPU 侧在关键推理参数一致的情况下仍保持相同 ACC，并且与第一轮 NPU 结果的归一化预测文本逐样本一致。

需要注意：

- 目前对齐的是 ACC 指标，不是逐样本原始输出文本的字节级一致。
- GPU 侧与 NPU 侧已补充关键推理参数对齐测试，但当前仍未拿到 GPU `details.csv` 与 NPU `results.csv` 做逐样本原始输出 diff。
- 如果要做更严格等价验证，应继续对齐两侧 `result.md` 原文、模型源码 commit、Transformers 小版本、随机种子和输出保存逻辑。
- 当前 prompt 是 `<image>document parsing.`，偏通用文档解析，不是专门为单词级 STR 优化的 prompt。

## 9. 已知限制

1. vLLM-Ascend 当前不支持 `UnlimitedOCRForCausalLM`，无法直接使用 vLLM serve。
2. NPU 侧依赖运行时 CUDA-to-NPU shim，属于工程兼容方案，不是官方原生 NPU 实现。
3. 当前 FastAPI 服务使用线程锁串行化请求，适合测试和低并发，不适合直接高并发生产。
4. 本次只测试 OCRBench 中 76 张经典 STR 抽样样本，不是完整 OCRBench benchmark。
5. `Unlimited-OCR` 官方主战场是长文档/整页解析，单词级 STR 结果不代表其文档 OCR SOTA 能力。
6. 部分算子可能落到 AiCPU，对极限性能有影响。

## 10. 复现命令

### 10.1 NPU 启动服务

```bash
docker exec -d unlimited-ocr-npu-test bash -lc \
  'ASCEND_RT_VISIBLE_DEVICES=8 \
   UNLIMITED_OCR_MODEL=/mnt/model/baidu/Unlimited-OCR \
   uvicorn serve_npu:app \
   --host 0.0.0.0 \
   --port 10080 \
   --app-dir /mnt/model/baidu/Unlimited-OCR \
   > /mnt/model/baidu/Unlimited-OCR/server.log 2>&1'
```

### 10.2 NPU 健康检查

```bash
curl -s http://127.0.0.1:10080/health
```

示例返回：

```json
{"status":"ok","npu_free":55720800256,"npu_total":65787658240}
```

### 10.3 NPU OCRBench 测试

基础服务参数测试：

```bash
python3 /mnt/model/baidu/Unlimited-OCR/benchmarks/run_ocrbench_str.py
```

GPU 参数对齐测试：

```bash
python3 /mnt/model/baidu/Unlimited-OCR/benchmarks/run_ocrbench_str_gpu_params.py
```

查看结果：

```bash
cat /mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval/summary.json
cat /mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval_gpu_params_npu/summary.json
```

### 10.4 GPU OCRBench 测试

在 Windows GPU 环境的 `D:\ch\Unlimited-OCR` 下执行：

```powershell
.\.venv\Scripts\python.exe eval_ocrbench_scene_text.py
```

GPU 侧模型路径：

```text
D:\ch\Unlimited-OCR\models\Unlimited-OCR
```

GPU 侧 OCRBench parquet：

```text
D:\ch\Unlimited-OCR\datasets\OCRBench\test-00000-of-00001.parquet
```

## 11. 最终结论

本次适配已证明：

- `Unlimited-OCR` 可以在 Ascend NPU 上通过 `Transformers + torch_npu` 路线完成推理。
- 当前 NPU FastAPI 服务可用，支持健康检查和 OCR 推理。
- OCRBench 经典场景文字识别子集上，GPU 与 NPU ACC 一致，均为 `64.47%`。
- NPU 侧补充 GPU 参数对齐测试后，ACC 仍为 `64.47%`，与第一轮 NPU 结果的逐样本归一化预测文本完全一致。
- vLLM-Ascend 暂不能直接 serve 该模型，生产高并发部署仍需进一步做 vLLM 模型注册适配或等待上游支持。

因此，当前可交付方案为：

```text
Transformers + torch_npu + FastAPI
```

适用场景：

- 功能验证
- 低并发内部 OCR 服务
- NPU 适配可行性测试
- 后续文档 OCR benchmark 的基础服务

不建议直接用于：

- 高并发生产服务
- vLLM 原生在线推理
- 单词级 STR SOTA 对比结论
