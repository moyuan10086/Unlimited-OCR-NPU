import base64
import os
import tempfile
import threading
import time
from typing import Optional

import torch
import torch_npu  # noqa: F401
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = os.environ.get("UNLIMITED_OCR_MODEL", "/mnt/model/baidu/Unlimited-OCR")
OUTPUT_DIR = os.environ.get("UNLIMITED_OCR_OUTPUT", os.path.join(MODEL_PATH, "server_output"))


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


_patch_cuda_calls()
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_safetensors=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).eval().npu()
torch.npu.synchronize()

app = FastAPI(title="Unlimited-OCR NPU")
lock = threading.Lock()


@app.get("/health")
def health():
    free, total = torch.npu.mem_get_info()
    return {"status": "ok", "npu_free": free, "npu_total": total}


@app.post("/ocr")
def ocr(req: OCRRequest):
    image_path = req.image_path
    temp_path = None
    if req.image_base64:
        data = req.image_base64
        if "," in data:
            data = data.split(",", 1)[1]
        fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="unlimited_ocr_")
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(data))
        image_path = temp_path

    if not image_path:
        return {"error": "image_path or image_base64 is required"}

    started = time.time()
    with lock:
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
        torch.npu.synchronize()

    if temp_path:
        os.unlink(temp_path)

    return {"text": text, "seconds": round(time.time() - started, 3)}
