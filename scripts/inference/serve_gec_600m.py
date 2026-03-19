"""
Fast GEC server — 600M single-tag model.
Port 15128. Used by "Тез тексеру" button.
Single <грамматика> tag, one model call, no pipeline.

Deploy: CUDA_VISIBLE_DEVICES=0 python serve_gec_600m.py --port 15128
"""
import argparse
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_ID = "stukenov/sozkz-core-llama-600m-kk-gec-v1"
TOKENIZER_ID = "stukenov/sozkz-core-llama-600m-kk-gec-v1"

model = None
tokenizer = None
prefix_kv = None  # Precomputed KV cache for "<грамматика> " prefix
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, prefix_kv
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast
    print(f"Loading model: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except (ValueError, ImportError):
        print("AutoTokenizer failed, using GPT2TokenizerFast fallback...")
        from huggingface_hub import hf_hub_download
        tok_file = hf_hub_download(repo_id=TOKENIZER_ID, filename="tokenizer.json")
        tokenizer = GPT2TokenizerFast(tokenizer_file=tok_file)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2, fall back to sdpa
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn_impl
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {params:.0f}M params on {device}, attn={attn_impl}")

    # torch.compile for kernel fusion
    if device == "cuda":
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile applied (reduce-overhead)")

        # Warmup: 3 passes to fully compile all paths
        print("Running warmup inference...")
        for i in range(3):
            _w = tokenizer("warmup text", return_tensors="pt", truncation=True, max_length=32)
            _w.pop("token_type_ids", None)
            _w = {k: v.to(device) for k, v in _w.items()}
            with torch.no_grad():
                model.generate(**_w, max_new_tokens=16, pad_token_id=tokenizer.eos_token_id)
        print("Warmup done (3 passes)")
    yield
    del model, tokenizer


app = FastAPI(title="SozKZ GEC 600M Fast", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "sozkz-gec-600m"
    messages: list[Message]
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512
    repeat_penalty: float = 1.1


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b): return _edit_distance(b, a)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _is_valid(original: str, corrected: str) -> bool:
    if not corrected or len(corrected) < 2:
        return False
    cyrillic = sum(1 for c in corrected if '\u0400' <= c <= '\u04FF')
    non_space = len(corrected.replace(' ', ''))
    if non_space > 3 and cyrillic / non_space < 0.8:
        return False
    ow, cw = original.split(), corrected.split()
    if abs(len(ow) - len(cw)) > 1:
        return False
    if _edit_distance(original, corrected) > max(3, int(len(original) * 0.2)):
        return False
    if ow and cw and ow[0][0].lower() != cw[0][0].lower():
        return False
    return True


def correct(text: str) -> str:
    """Single-tag correction: one model call."""
    prompt = f"<грамматика> {text}\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = len(inputs["input_ids"][0])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=min(input_len + 30, 200),
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    if "\u2192 " in result:
        result = result.split("\u2192 ", 1)[1]
    for stop in ["\n<", "\n\n"]:
        if stop in result:
            result = result[:result.index(stop)]
    result = result.strip()
    while result and not result[0].isalpha() and result[0] not in "({[\"'":
        result = result[1:]
    result = result.strip()
    if result and result != text and _is_valid(text, result):
        return result
    return text


@app.post("/v1/gec/completions")
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    t0 = time.time()
    user_text = ""
    for msg in req.messages:
        if msg.role == "user":
            user_text = msg.content.strip()
    if not user_text:
        return {"error": "No user message found"}

    async def _do_inference():
        async with inference_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, correct, user_text)

    try:
        corrected = await asyncio.wait_for(_do_inference(), timeout=15.0)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=503, content={
            "error": "Server busy",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": user_text}, "finish_reason": "timeout"}],
        })

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "sozkz-gec-600m-fast",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": corrected}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(user_text.split()), "completion_tokens": len(corrected.split()), "total_tokens": len(user_text.split()) + len(corrected.split())},
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=15128)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
