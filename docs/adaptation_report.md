# Unlimited-OCR Ascend NPU 适配测试报告

## 1. 测试结论

本次在 Ascend NPU 环境完成了 Baidu `Unlimited-OCR` 模型的部署与推理适配测试。模型可通过 `Transformers + torch_npu` 路径在 Ascend 910 上完成 OCR 推理，HTTP 服务接口验证通过。

当前结论：

- `Transformers + torch_npu`：可用，已完成模型加载、NPU 推理、HTTP 接口验证。
- `vLLM-Ascend`：当前不可直接使用，原因是现有 vLLM-Ascend 镜像不支持模型架构 `UnlimitedOCRForCausalLM`。
- 当前部署方式适合功能验证、低并发测试、内部工具化调用。
- 若用于生产高并发服务，需要进一步做 vLLM-Ascend 原生适配或等待上游支持。

## 2. 测试环境

测试时间：2026-07-03 15:52:46 CST

硬件与系统：

- 操作系统：openEuler 22.03 LTS-SP4，aarch64
- 加速卡：Ascend 910
- NPU 状态：主机可见 8 张 Ascend 910，每张含 2 个 chip，共 16 个可见 chip
- 本次测试使用空闲物理 chip：`ASCEND_RT_VISIBLE_DEVICES=8`

容器与软件：

- 测试容器：`unlimited-ocr-npu-test`
- 基础镜像：`quay.io/ascend/vllm-ascend:v0.18.0-a3`
- vLLM：`0.18.0`
- vLLM-Ascend：`0.18.0`
- PyTorch：`2.9.0+cpu`
- torch_npu：`2.9.0.post1`
- Transformers：`4.57.6`
- Python：容器内 Python 3.11

模型文件：

- 模型目录：`/mnt/model/baidu/Unlimited-OCR`
- 权重文件：`model-00001-of-000001.safetensors`
- 权重大小：约 6.3 GB
- 模型架构：`UnlimitedOCRForCausalLM`
- 模型类型：`unlimited-ocr`

## 3. 部署方式

由于模型官方示例主要面向 CUDA/NVIDIA GPU，模型源码中存在多处 `.cuda()` 和 `torch.autocast("cuda")` 调用。Ascend NPU 环境无法直接执行这些 CUDA 调用。

本次采用运行时 shim 的方式完成兼容：

- 将 `torch.Tensor.cuda()` 映射为 `tensor.to("npu")`
- 将 `torch.nn.Module.cuda()` 映射为 `module.to("npu")`
- 将 `torch.autocast("cuda")` 映射为 `torch.autocast("npu")`

该方式没有修改官方模型源码，仅在服务启动时做运行时兼容处理。服务脚本位置：

```text
/mnt/model/baidu/Unlimited-OCR/serve_npu.py
```

服务启动方式：

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

服务地址：

```text
http://127.0.0.1:10080
```

接口：

- `GET /health`：健康检查与 NPU 显存信息
- `POST /ocr`：OCR 推理接口

## 4. 验证结果

### 4.1 模型加载验证

模型可通过 `AutoTokenizer` 与 `AutoModel.from_pretrained(..., trust_remote_code=True)` 正常加载。

加载到空闲 NPU chip 8 后，模型参数设备为 `npu:0`。测试时 NPU HBM 信息如下：

```text
加载前 free/total: 65384194048 / 65787658240
加载后 free/total: 58448732160 / 65787658240
```

模型加载后约占用 6.9 GB HBM。

### 4.2 HTTP 服务验证

健康检查结果：

```json
{
  "status": "ok",
  "npu_free": 58449616896,
  "npu_total": 65787658240
}
```

测试请求：

```bash
curl -s -X POST http://127.0.0.1:10080/ocr \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"/mnt/model/baidu/Unlimited-OCR/test_input.png","max_length":4096}'
```

测试输出：

```text
<|det|>text [59, 194, 273, 240]<|/det|>Invoice No: UOCR-2026-0703
<|det|>text [59, 396, 190, 443]<|/det|>Total: 123.45 USD
<|det|>text [59, 608, 255, 656]<|/det|>Deploy test on Ascend NPU
```

单张测试图推理耗时约 3.44 秒。

### 4.3 日志情况

服务日志位置：

```text
/mnt/model/baidu/Unlimited-OCR/server.log
```

日志中存在以下提示：

- `model.vision_model.embeddings.position_ids` 未从 checkpoint 初始化。该字段通常属于位置 ID 缓冲/辅助参数，不影响本次 OCR 功能验证结果。
- 推理时提示 attention mask 未设置，并自动将 `pad_token_id` 设为 `eos_token_id`。这是 Transformers 生成接口的告警，当前模型官方 `infer` 方法未显式传入 attention mask。
- `ArgSort` 部分算子在 AiCPU 上执行，说明该算子未走 AiCore，可能影响极限性能，但不影响功能正确性。

## 5. vLLM-Ascend 适配情况

尝试使用 vLLM-Ascend 启动服务：

```bash
ASCEND_RT_VISIBLE_DEVICES=8 vllm serve /mnt/model/baidu/Unlimited-OCR \
  --served-model-name Unlimited-OCR \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 10080 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.5 \
  --dtype bfloat16
```

启动失败，核心报错：

```text
Model architectures ['UnlimitedOCRForCausalLM'] are not supported for now.
```

原因分析：

- `Unlimited-OCR` 使用自定义模型架构 `UnlimitedOCRForCausalLM`。
- 当前 `vllm-ascend:v0.18.0-a3` 内置支持的 OCR/VL 模型包括 `DeepseekOCRForCausalLM`、`DeepseekOCR2ForCausalLM`、`GlmOcrForConditionalGeneration` 等，但不包含 `UnlimitedOCRForCausalLM`。
- 即使使用 `--trust-remote-code`，vLLM 仍需要模型架构在其模型注册表中受支持，不能像 Transformers 一样直接加载任意 remote code 模型。

因此，当前不能通过 vLLM-Ascend 原生方式直接部署该模型。

## 6. 影响评估

### 6.1 对当前测试的影响

无阻塞影响。模型已经通过 Transformers + torch_npu 路径完成 NPU 推理验证，HTTP 接口可正常使用。

### 6.2 对性能的影响

有一定影响。

当前服务是单模型直接推理，并通过线程锁控制并发，避免多个请求同时竞争 NPU 显存。因此：

- 适合功能测试、低并发调用、内部验证。
- 不适合直接作为高并发生产服务。
- 相比 vLLM，缺少连续批处理、KV cache 管理、调度优化等服务化能力。

### 6.3 对稳定性的影响

中等风险。

当前适配依赖运行时 shim，将 CUDA 调用转到 NPU。该方式已通过本次验证，但仍属于兼容适配，不是官方原生 NPU 支持。后续如果升级模型代码、Transformers、torch_npu、CANN 或容器镜像，需要重新回归验证。

### 6.4 对生产部署的影响

如仅用于低频 OCR 任务，可以继续使用当前 FastAPI 服务方式。

如用于生产高并发场景，建议：

- 做正式 vLLM-Ascend 模型适配，将 `UnlimitedOCRForCausalLM` 注册进 vLLM 模型体系；
- 或等待 vLLM-Ascend / vLLM 上游支持该模型架构；
- 或评估 SGLang/NPU 侧是否存在更成熟适配路径；
- 或将当前 Transformers 服务封装为受控队列服务，限制并发并做好超时、监控、重启策略。

## 7. 当前遗留问题

1. 官方模型代码硬编码 CUDA 调用，当前通过运行时 shim 绕过。
2. vLLM-Ascend 尚不支持 `UnlimitedOCRForCausalLM`。
3. 目前仅验证单图 OCR，未验证多页 PDF、长图、高分辨率图和批量请求。
4. 当前服务为单进程 FastAPI，未做生产级鉴权、限流、队列、监控和自动重启。
5. 部分算子落到 AiCPU，后续需做性能 profiling。

## 8. 后续建议

短期建议：

- 保留当前 `Transformers + torch_npu + FastAPI` 方案作为可用测试服务。
- 将 `ASCEND_RT_VISIBLE_DEVICES` 固定到空闲 chip，例如当前验证可用的 `8`。
- 对服务增加 systemd、supervisor 或容器启动脚本，避免手动启动。
- 增加真实业务样本测试，包括表格、票据、长图、多页文档。

中期建议：

- 将 shim 逻辑整理为正式适配层，避免散落在业务代码中。
- 增加队列化请求处理，限制并发，减少 NPU OOM 风险。
- 补充日志、耗时、失败率、HBM 使用率监控。

长期建议：

- 推进 vLLM-Ascend 原生适配 `UnlimitedOCRForCausalLM`。
- 对模型结构、视觉编码器、投影层、DeepseekV2 语言模型部分进行 vLLM 模型注册和权重加载适配。
- 验证 vLLM 下的多模态输入、custom logits processor、长上下文和连续批处理能力。

## 9. 最终结论

`Unlimited-OCR` 在本机 Ascend NPU 上可以运行，当前已完成模型下载、依赖安装、NPU 加载、单图 OCR 推理和 HTTP 服务验证。

当前可交付部署方式为：

```text
Transformers + torch_npu + FastAPI
```

当前不支持的部署方式为：

```text
vLLM-Ascend 原生 serve
```

对当前功能测试无影响；对生产高并发部署有影响，需要进一步做 vLLM-Ascend 原生适配或采用受控队列化的 Transformers 服务方案。
