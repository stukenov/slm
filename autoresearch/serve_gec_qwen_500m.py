"""
FastAPI inference server for GEC Qwen 500M model.
OpenAI-compatible /v1/gec/completions endpoint.

Pipeline: input → емле fixer (dictionary) → model GEC → емле post-fix → output

Deploy on kaznu: python serve_gec_qwen_500m.py --port 15131
"""
import argparse
import asyncio
import json
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_ID = "stukenov/sozkz-fix-qwen-500m-kk-gec-v3"
INSTRUCTION = (
    "Мәтіндегі грамматикалық, орфографиялық, пунктуациялық және сөз қолданысындағы "
    "қателерді түзет. Мағынаны өзгертпе. Егер мәтін дұрыс болса, оны өзгеріссіз қайтар. "
    "Тек түзетілген мәтінді қайтар."
)

model = None
tokenizer = None
full_dict = {}
kz_dict = {}
inference_lock = asyncio.Lock()

REVERSE_MAP = {
    "у": ["ү", "ұ"], "о": ["ө"], "к": ["қ"], "н": ["ң"],
    "г": ["ғ"], "а": ["ә"], "и": ["і"], "х": ["һ"],
}


def load_emle_dicts():
    from itertools import product as _p
    fd, kd = {}, {}
    for path, target in [("/root/kz_full_dict.json", "full"), ("/root/kz_word_freq.json", "kz")]:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if target == "full":
                fd = data
            else:
                kd = data
            print(f"Емле {target} dict: {len(data)} words")
        else:
            print(f"WARNING: {path} not found")
    return fd, kd


def _find_correction(word, full_dict, kz_dict):
    from itertools import product
    lower = word.lower()
    orig_freq = full_dict.get(lower, 0)
    if lower in kz_dict and kz_dict[lower] >= 50:
        return None
    positions = []
    for i, ch in enumerate(lower):
        if ch in REVERSE_MAP:
            positions.append((i, [ch] + REVERSE_MAP[ch]))
    if not positions or len(positions) > 8:
        return None
    best, best_freq = None, 0
    for combo in product(*[opts for _, opts in positions]):
        candidate = list(lower)
        changed = False
        for (idx, _), repl in zip(positions, combo):
            if repl != lower[idx]:
                changed = True
            candidate[idx] = repl
        if not changed:
            continue
        cs = "".join(candidate)
        if cs in kz_dict and kz_dict[cs] > best_freq:
            best, best_freq = cs, kz_dict[cs]
    if best is None:
        return None
    if orig_freq == 0:
        return best if best_freq >= 3 else None
    return best if best_freq / max(orig_freq, 1) > 5 else None


def fix_emle(text, full_dict, kz_dict):
    if not full_dict:
        return text

    def replace_word(m):
        word = m.group(0)
        correction = _find_correction(word, full_dict, kz_dict)
        if correction:
            if word[0].isupper() and not word.isupper():
                correction = correction[0].upper() + correction[1:]
            elif word.isupper():
                correction = correction.upper()
            return correction
        return word

    return re.sub(r"[а-яәғқңөұүһіА-ЯӘҒҚҢӨҰҮҺІ]+", replace_word, text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, full_dict, kz_dict
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    from huggingface_hub import hf_hub_download

    token = None
    for p in ["~/.cache/huggingface/token", "/root/.HUGGINGFACE_HUB_TOKEN"]:
        expanded = os.path.expanduser(p)
        if os.path.exists(expanded):
            token = open(expanded).read().strip()
            break

    full_dict, kz_dict = load_emle_dicts()

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, token=token,
    )
    tok_file = hf_hub_download(MODEL_ID, "tokenizer.json", token=token)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {params:.0f}M params on {device}")

    if device == "cuda":
        print("Running warmup inference...")
        _ids = tokenizer.encode("warmup test", return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(_ids, max_new_tokens=16, pad_token_id=1)
        print("Warmup done")
    yield
    del model, tokenizer


app = FastAPI(title="SozKZ GEC Qwen 500M", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "sozkz-gec-qwen-500m"
    messages: list[Message]
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512


def _edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
        prev = curr
    return prev[lb]


def correct(text: str) -> str:
    original_ends_with_punct = text.rstrip().endswith(('.', '!', '?', '…'))
    text = fix_emle(text, full_dict, kz_dict)
    prompt = f"### Нұсқау:\n{INSTRUCTION}\n\n### Мәтін:\n{text}\n\n### Түзетілген:\n"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(ids)
    input_len = ids.shape[1]
    num_beams = 4
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=min(input_len, 512),
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=1,
        )
    candidates = []
    for seq in out:
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        c = decoded.split("### Түзетілген:\n")[-1].split("###")[0].strip()
        if c:
            candidates.append(c)
    if not candidates:
        return text
    # Rerank: prefer candidate closest to input (minimal edit distance)
    best = min(candidates, key=lambda c: _edit_distance(text, c))
    best = fix_emle(best, full_dict, kz_dict)
    if not original_ends_with_punct and best.endswith('.'):
        best = best[:-1].rstrip()
    return best if best else text


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
        corrected = await asyncio.wait_for(_do_inference(), timeout=30.0)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=503, content={
            "error": "Server busy",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": user_text}, "finish_reason": "timeout"}],
        })
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "error": "Inference error",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": user_text}, "finish_reason": "error"}],
        })

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "sozkz-gec-qwen-500m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": corrected}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(user_text.split()), "completion_tokens": len(corrected.split()), "total_tokens": len(user_text.split()) + len(corrected.split())},
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=15131)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
