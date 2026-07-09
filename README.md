# Unlimited-OCR Ascend NPU Adaptation

[中文说明](README.zh-CN.md)

This repository contains an Ascend NPU adaptation and benchmark harness for
`baidu/Unlimited-OCR`.

It keeps the upstream model code required by `trust_remote_code=True`, plus:

- `serve_npu.py`: FastAPI service for Transformers + `torch_npu` inference.
- `benchmarks/run_ocrbench_str.py`: OCRBench STR sanity test.
- `benchmarks/run_ocrbench_str_gpu_params.py`: OCRBench STR test using the GPU-side inference parameters.
- `docs/`: deployment, benchmark, GPU/NPU alignment, and SOTA notes.

Large files are intentionally not committed:

- model weights, especially `model-00001-of-000001.safetensors`
- OCRBench parquet files
- generated benchmark images/results
- logs, caches, and server output directories

## Tested Conclusion

On Ascend 910, `Unlimited-OCR` can run through:

```text
Transformers + torch_npu + FastAPI
```

`vLLM-Ascend` cannot directly serve this model in the tested environment because
`UnlimitedOCRForCausalLM` is not supported by the vLLM-Ascend model registry.

OCRBench classic scene text recognition subset:

```text
49 / 76 = 64.47%
```

The NPU result matches the GPU-side ACC on the same 76 samples. A second NPU run
using GPU-side inference parameters also produced `49 / 76 = 64.47%`.

## Model Files

Download the official model files from:

- https://huggingface.co/baidu/Unlimited-OCR

Place the model weights in the repository root, for example:

```text
model-00001-of-000001.safetensors
model.safetensors.index.json
```

The large `.safetensors` file is ignored by git.

## NPU Service

Example startup inside the Ascend test container:

```bash
ASCEND_RT_VISIBLE_DEVICES=8 \
UNLIMITED_OCR_MODEL=/mnt/model/baidu/Unlimited-OCR \
uvicorn serve_npu:app \
  --host 0.0.0.0 \
  --port 10080 \
  --app-dir /mnt/model/baidu/Unlimited-OCR
```

Health check:

```bash
curl -s http://127.0.0.1:10080/health
```

OCR request:

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

## OCRBench Test

Download OCRBench parquet:

```text
https://huggingface.co/datasets/echo840/OCRBench/resolve/main/data/test-00000-of-00001.parquet
```

Expected local path used by the scripts:

```text
/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/test-00000-of-00001.parquet
```

Run the base-parameter NPU benchmark:

```bash
python3 benchmarks/run_ocrbench_str.py
```

Run the GPU-parameter-aligned NPU benchmark:

```bash
python3 benchmarks/run_ocrbench_str_gpu_params.py
```

## Reports

- `docs/adaptation_report.md`
- `docs/gpu_npu_adaptation_report.md`
- `docs/ocrbench_str_report.md`
- `docs/sota_comparison.md`
- `docs/UPSTREAM_README.md`

## Notes

This repository is an engineering adaptation package. It is not an official
Baidu release. The upstream license is kept in `LICENSE`.
