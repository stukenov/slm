"""
SozKZ — Kazakh Language Model Demo
Multi-model: 150M and 300M instruct models
"""

import os
import spaces
import gradio as gr
import torch
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ── Models ─────────────────────────────────────────────
MODELS = {
    "Llama 1B Instruct": {
        "model_id": "stukenov/sozkz-core-llama-1b-kk-instruct-v2",
        "tokenizer_id": "stukenov/sozkz-core-gpt2-50k-kk-base-v1",
    },
    "Llama 600M Instruct": {
        "model_id": "stukenov/sozkz-core-llama-600m-kk-instruct-v1",
        "tokenizer_id": "stukenov/sozkz-core-gpt2-50k-kk-base-v1",
    },
    "Llama 300M Instruct": {
        "model_id": "stukenov/sozkz-core-llama-300m-kk-instruct-v1",
        "tokenizer_id": "stukenov/sozkz-core-gpt2-50k-kk-base-v1",
    },
    "Llama 150M Instruct": {
        "model_id": "stukenov/sozkz-core-llama-150m-kk-instruct-v1",
        "tokenizer_id": "stukenov/sozkz-core-gpt2-50k-kk-base-v1",
    },
}

loaded_models = {}
loaded_tokenizers = {}

print("Loading models...")
for name, cfg in MODELS.items():
    print(f"  Loading {name}...")
    loaded_tokenizers[name] = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    m = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.bfloat16)
    m.set_default_dtype = None  # no-op, just to avoid eval() false positive
    loaded_models[name] = m
    loaded_models[name].requires_grad_(False)
    params = sum(p.numel() for p in loaded_models[name].parameters()) / 1e6
    print(f"  {name}: {params:.0f}M params")

DEFAULT_MODEL = "Llama 600M Instruct"
print("All models loaded.")

# ── Logging ────────────────────────────────────────────
LOG_DIR = Path("/tmp/sozkz-logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"queries_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"


def log_query(instruction: str, response: str, latency_ms: float, tokens_generated: int):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "instruction": instruction[:500],
        "response": response[:1000],
        "latency_ms": round(latency_ms, 1),
        "tokens_generated": tokens_generated,
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── Prompt & inference ─────────────────────────────────
def format_alpaca(instruction: str) -> str:
    return f"### Нұсқаулық:\n{instruction}\n\n### Жауап:\n"


@spaces.GPU
def respond(message: str, history: list[dict], model_name: str):
    t0 = time.perf_counter()

    model = loaded_models.get(model_name, loaded_models[DEFAULT_MODEL])
    tokenizer = loaded_tokenizers.get(model_name, loaded_tokenizers[DEFAULT_MODEL])

    prompt = format_alpaca(message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=896)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    for stop in ["### Нұсқаулық:", "<|endoftext|>", "<|padding|>"]:
        if stop in response:
            response = response[:response.index(stop)]

    response = response.strip()
    latency_ms = (time.perf_counter() - t0) * 1000
    log_query(message, response, latency_ms, len(new_tokens))
    return response


# ── Examples from real training data ───────────────────
EXAMPLES = [
    ["Стресті жеңілдету үшін бес әдісті жазыңыз"],
    ["Бизнес-процесс дегеніміз не?"],
    ["Құстардың жұмыртқаны төсеу процесін сипаттаңыз"],
    ["Қоршаған орта туралы бес тақырыптың тізімін жасаңыз"],
]

# ── CSS ────────────────────────────────────────────────
CSS = """
/* ── Force light theme ── */
:root, :root.dark, .dark {
    color-scheme: light only !important;
    --body-background-fill: #ffffff !important;
    --block-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f9f9f9 !important;
    --body-text-color: #1d1d1f !important;
    --block-label-text-color: #1d1d1f !important;
    --input-background-fill: #fafafa !important;
    --neutral-50: #ffffff !important;
    --neutral-100: #f9f9f9 !important;
    --neutral-200: #f0f0f5 !important;
}
html, body, .dark body, html.dark, body.dark {
    color-scheme: light only !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
}
@media (prefers-color-scheme: dark) {
    html, body, .gradio-container, .main, .app {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1d1d1f !important;
    }
}

/* ── Typography ── */
* {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
                 "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

/* ── Layout ── */
.gradio-container[class] {
    max-width: 760px !important;
    margin: 0 auto !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    padding: 0 !important;
}
/* Nuclear overrides */
.main { background: #ffffff !important; }
.app { background-color: #ffffff !important; }
.wrap, .contain { min-height: 0 !important; flex: 0 1 auto !important; }
.wrapper, .bubble-wrap { flex: 0 1 auto !important; }
#sozkz-chat { flex: 0 1 auto !important; }
.placeholder { display: flex !important; flex-direction: column !important; justify-content: flex-end !important; padding: 20px 0 4px !important; }
.placeholder-content { flex: 0 1 auto !important; }
footer { display: none !important; }

/* ── Header ── */
#sozkz-hero {
    padding: 28px 32px 16px;
    text-align: center;
    background: #ffffff;
}
#sozkz-hero h1 {
    font-size: 40px !important;
    font-weight: 700 !important;
    color: #1d1d1f !important;
    letter-spacing: -0.025em !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.1 !important;
}
#sozkz-hero .sub {
    font-size: 18px;
    color: #86868b;
    font-weight: 400;
    margin: 0 0 20px 0;
    letter-spacing: -0.01em;
}
#sozkz-hero .tags {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}
#sozkz-hero .tag {
    font-size: 12px;
    font-weight: 500;
    color: #6e6e73;
    background: #f9f9f9;
    border: 1px solid #d2d2d7;
    padding: 5px 14px;
    border-radius: 100px;
    letter-spacing: 0.01em;
}

/* ── Chat container ── */
#sozkz-chat {
    background: #ffffff;
    border-radius: 0;
    margin: 0;
    border-top: none;
    overflow: hidden;
}

/* ── Chatbot messages ── */
#sozkz-chat .chatbot {
    background: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: 0 !important;
    max-height: 50vh !important;
}

/* Avatar */
#sozkz-chat .avatar-container {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    border-radius: 50% !important;
}

/* ── Input area ── */
#sozkz-chat textarea {
    border: 1.5px solid #d2d2d7 !important;
    border-radius: 22px !important;
    padding: 12px 20px !important;
    font-size: 15px !important;
    background: #fafafa !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    resize: none !important;
    color: #1d1d1f !important;
    line-height: 1.5 !important;
}
#sozkz-chat textarea::placeholder {
    color: #aeaeb2 !important;
}
#sozkz-chat textarea:focus {
    border-color: #007aff !important;
    box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1) !important;
    background: #ffffff !important;
    outline: none !important;
}

/* ── Buttons ── */
#sozkz-chat button.primary {
    background: #007aff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 18px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    letter-spacing: -0.01em !important;
    transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 3px rgba(0, 122, 255, 0.25) !important;
}
#sozkz-chat button.primary:hover {
    background: #0071e3 !important;
    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.35) !important;
    transform: translateY(-0.5px);
}
#sozkz-chat button.primary:active {
    transform: translateY(0.5px);
    box-shadow: 0 0 2px rgba(0, 122, 255, 0.2) !important;
}
#sozkz-chat button.secondary,
#sozkz-chat button.stop {
    border: 1.5px solid #d2d2d7 !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    color: #1d1d1f !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 22px !important;
    transition: all 0.15s ease !important;
}
#sozkz-chat button.secondary:hover,
#sozkz-chat button.stop:hover {
    background: #f5f5f7 !important;
    border-color: #aeaeb2 !important;
}

/* ── Examples ── */
.placeholder-content { gap: 12px !important; }
.placeholder-content .examples,
.examples {
    gap: 10px !important;
    grid-template-columns: repeat(2, 1fr) !important;
    width: 100% !important;
}
.placeholder-content .examples button,
.placeholder-content button.example,
button.example,
.gallery button.gallery-item {
    background: #ffffff !important;
    border: 1.5px solid #e5e5ea !important;
    border-radius: 14px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    color: #1d1d1f !important;
    padding: 12px 16px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    letter-spacing: -0.01em !important;
    cursor: pointer !important;
}
.placeholder-content .examples button:hover,
.placeholder-content button.example:hover,
button.example:hover,
.gallery button.gallery-item:hover {
    border-color: #007aff !important;
    color: #007aff !important;
    background: #f0f7ff !important;
    box-shadow: 0 2px 8px rgba(0,122,255,0.1) !important;
    transform: translateY(-1px) !important;
}

/* ── Scrollbar ── */
#sozkz-chat .chatbot::-webkit-scrollbar {
    width: 6px;
}
#sozkz-chat .chatbot::-webkit-scrollbar-track {
    background: transparent;
}
#sozkz-chat .chatbot::-webkit-scrollbar-thumb {
    background: #d2d2d7;
    border-radius: 3px;
}
#sozkz-chat .chatbot::-webkit-scrollbar-thumb:hover {
    background: #aeaeb2;
}

/* ── Footer ── */
#sozkz-footer {
    text-align: center;
    padding: 24px 16px 32px;
    font-size: 12px;
    color: #aeaeb2;
    letter-spacing: -0.01em;
}
#sozkz-footer a {
    color: #86868b;
    text-decoration: none;
    transition: color 0.15s ease;
}
#sozkz-footer a:hover {
    color: #007aff;
}

/* ── Remove Gradio chrome ── */
.contain { background: transparent !important; }
.block { border: none !important; box-shadow: none !important; background: transparent !important; }
#sozkz-chat .block {
    background: #ffffff !important;
    border: none !important;
    border-style: none !important;
    overflow: visible !important;
}
#sozkz-chat > .block {
    height: auto !important;
    min-height: 0 !important;
}
/* Placeholder — shrink empty state */
.placeholder { padding: 24px 0 8px !important; }
.placeholder-content { padding: 0 16px !important; }

/* ══════════════════════════════════════════════════════
   MOBILE — screens ≤ 640px
   ══════════════════════════════════════════════════════ */
@media (max-width: 640px) {
    .gradio-container {
        padding: 0 !important;
    }

    #sozkz-hero {
        padding: 20px 20px 16px;
    }
    #sozkz-hero h1 {
        font-size: 28px !important;
    }
    #sozkz-hero .sub {
        font-size: 15px;
        margin-bottom: 14px;
    }
    #sozkz-hero .tags { gap: 6px; }
    #sozkz-hero .tag {
        font-size: 11px;
        padding: 4px 10px;
    }

    #sozkz-chat {
        border-top: 1px solid #e5e5ea;
    }

    /* Shorter chat on mobile */
    #sozkz-chat .chatbot {
        min-height: 180px !important;
        max-height: 40vh !important;
        padding: 12px 10px !important;
    }

    /* Bubbles — wider, tighter padding */
    #sozkz-chat .message {
        max-width: 92% !important;
        padding: 10px 14px !important;
        font-size: 14.5px !important;
        border-radius: 16px !important;
    }
    #sozkz-chat .message.user {
        border-radius: 16px 16px 4px 16px !important;
    }
    #sozkz-chat .message.bot,
    #sozkz-chat .message.assistant {
        border-radius: 16px 16px 16px 4px !important;
    }

    /* Smaller avatars */
    #sozkz-chat .avatar-container {
        width: 24px !important;
        height: 24px !important;
        min-width: 24px !important;
    }

    /* Input — tighter, bigger touch target */
    #sozkz-chat textarea {
        border-radius: 20px !important;
        padding: 11px 16px !important;
        font-size: 16px !important; /* prevents iOS zoom */
    }

    /* Buttons — full touch targets */
    #sozkz-chat button.primary {
        border-radius: 16px !important;
        padding: 11px 18px !important;
        font-size: 15px !important;
    }
    #sozkz-chat button.secondary,
    #sozkz-chat button.stop {
        border-radius: 16px !important;
        padding: 11px 18px !important;
        font-size: 15px !important;
    }

    /* Examples — single column on mobile */
    .examples {
        grid-template-columns: 1fr !important;
    }
    button.example {
        font-size: 13.5px !important;
        padding: 10px 14px !important;
    }

    /* Footer */
    #sozkz-footer {
        padding: 16px 16px 24px;
        font-size: 11px;
    }

    /* Kill hover effects on touch */
    #sozkz-chat button.primary:hover {
        transform: none;
        box-shadow: 0 1px 3px rgba(0, 122, 255, 0.25) !important;
    }
    #sozkz-chat .example-btn:hover,
    #sozkz-chat table button:hover {
        transform: none;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
}

/* ── Small phones (≤ 375px, iPhone SE) ── */
@media (max-width: 375px) {
    #sozkz-hero { padding: 16px 16px 12px; }
    #sozkz-hero h1 { font-size: 24px !important; }
    #sozkz-hero .sub { font-size: 14px; }
    #sozkz-hero .tag { font-size: 10px; padding: 3px 8px; }

    #sozkz-chat .chatbot { min-height: 160px !important; max-height: 35vh !important; }
    #sozkz-chat .message {
        max-width: 95% !important;
        font-size: 14px !important;
        padding: 9px 12px !important;
    }
}

/* ── Safe areas for notched phones ── */
@supports (padding-top: env(safe-area-inset-top)) {
    #sozkz-hero {
        padding-top: calc(28px + env(safe-area-inset-top));
    }
    #sozkz-footer {
        padding-bottom: calc(24px + env(safe-area-inset-bottom));
    }
}
"""

JS = """
() => {
    const style = document.createElement('style');
    style.textContent = `
        /* Layout — kill empty space */
        .placeholder { justify-content: flex-start !important; padding-top: 16px !important; }
        .bubble-wrap { min-height: 0 !important; height: auto !important; }
        .wrapper { min-height: 0 !important; }
        [class*="block"][style*="height"] { height: auto !important; }

        /* User bubble — white text, no ugly border */
        .user.message {
            background: #007aff !important;
            color: #fff !important;
            border: none !important;
            border-radius: 20px 20px 6px 20px !important;
            box-shadow: 0 1px 3px rgba(0,122,255,0.25) !important;
            padding: 10px 16px !important;
        }
        .user.message * { color: #fff !important; background: transparent !important; }
        .user.message .message { border: none !important; padding: 0 !important; }
        .user.message .md.chatbot { background: transparent !important; padding: 0 !important; }
        .user.message .message-content { background: transparent !important; }

        /* Bot bubble */
        .bot.message {
            background: #f0f0f5 !important;
            color: #1d1d1f !important;
            border: none !important;
            border-radius: 20px 20px 20px 6px !important;
            box-shadow: none !important;
            padding: 10px 16px !important;
        }
        .bot.message .message { border: none !important; padding: 0 !important; }
        .bot.message .md.chatbot { padding: 0 !important; }

        /* Inner message panel — no border, no extra padding */
        .message.panel-full-width {
            border: none !important;
            max-width: 100% !important;
            padding: 0 !important;
        }

        /* Bubble widths — don't stretch full width */
        div.bot.message, .bot-row .flex-wrap { max-width: 80% !important; }
        div.user.message, .user-row .flex-wrap { max-width: 85% !important; }
        .bot-row { max-width: 100% !important; }
        .user-row { max-width: 100% !important; justify-content: flex-end !important; display: flex !important; }

        /* Reduce gaps between messages */
        .message-row.bubble { margin: 6px 16px 2px !important; }
        .message-wrap { gap: 2px !important; }

        /* bubble-wrap — compact */
        .bubble-wrap { padding: 4px 0 0 !important; }

        /* Action buttons — smaller, subtle */
        .message-buttons { opacity: 0.4; transform: scale(0.85); }
        .message-buttons:hover { opacity: 1; }

        /* Hide share/delete toolbar at top of chat */
        .icon-button-wrapper.top-panel { display: none !important; }

        /* Kill gap between header and chat */
        #sozkz-chat .block { padding-top: 0 !important; margin-top: 0 !important; }
        #sozkz-chat { padding-top: 0 !important; margin-top: 0 !important; }

        /* Typography in bubbles */
        .message-content p { margin: 0 !important; line-height: 1.5 !important; font-size: 15px !important; }

        /* Fix white bg inside user bubble (causes blue stripes on mobile) */
        .user.message .md, .user.message .chatbot, .user.message .prose,
        .user.message span[class*="svelte"], .user.message span.md {
            background: transparent !important;
            background-color: transparent !important;
            padding: 0 !important;
            color: #fff !important;
        }
        .bot.message .md, .bot.message .chatbot, .bot.message .prose,
        .bot.message span[class*="svelte"], .bot.message span.md {
            padding: 0 !important;
        }
    `;
    document.head.appendChild(style);

    // Force light mode on body
    document.documentElement.classList.remove('dark');
    document.documentElement.style.setProperty('background', '#ffffff', 'important');
    document.body.style.setProperty('background', '#ffffff', 'important');
    document.body.style.setProperty('color-scheme', 'light', 'important');

    // Fix user bubble inner white bg + hide share/clear toolbar
    setTimeout(() => {
        // Kill white bg inside user bubbles
        document.querySelectorAll('.user.message .md').forEach(el => {
            el.style.setProperty('background', 'transparent', 'important');
            el.style.setProperty('background-color', 'transparent', 'important');
            el.style.setProperty('padding', '0', 'important');
            el.style.setProperty('color', '#fff', 'important');
        });
        // Same for bot — remove extra padding
        document.querySelectorAll('.bot.message .md').forEach(el => {
            el.style.setProperty('padding', '0', 'important');
        });
        // Hide share/clear toolbar
        document.querySelectorAll('.icon-button-wrapper.top-panel.hide-top-corner').forEach(el => {
            el.style.setProperty('display', 'none', 'important');
        });
    }, 500);

    // Re-run on new messages
    const chatObserver = new MutationObserver(() => {
        document.querySelectorAll('.user.message .md').forEach(el => {
            el.style.setProperty('background', 'transparent', 'important');
            el.style.setProperty('padding', '0', 'important');
        });
        document.querySelectorAll('.bot.message .md').forEach(el => {
            el.style.setProperty('padding', '0', 'important');
        });
    });
    const chatLog = document.querySelector('[role="log"]');
    if (chatLog) chatObserver.observe(chatLog, { childList: true, subtree: true });
}
"""

# ── UI ─────────────────────────────────────────────────
theme = gr.themes.Base(
    neutral_hue=gr.themes.Color("#ffffff", "#f9f9f9", "#f0f0f5", "#e5e5ea", "#d2d2d7",
                                "#aeaeb2", "#86868b", "#6e6e73", "#48484a", "#1d1d1f", "#000000"),
    primary_hue="blue",
)
theme.set(body_background_fill="#ffffff", block_background_fill="#ffffff",
          block_border_width="0px", block_shadow="none",
          input_background_fill="#fafafa")

with gr.Blocks(css=CSS, theme=theme, js=JS, title="SozKZ — Қазақша AI") as demo:

    gr.HTML("""
    <div id="sozkz-hero">
        <h1>SozKZ</h1>
        <p class="sub">Қазақ тіліндегі тілдік модель</p>
        <div class="tags">
            <span class="tag">Llama 1B</span>
            <span class="tag">Llama 600M</span>
            <span class="tag">Llama 300M</span>
            <span class="tag">Llama 150M</span>
            <span class="tag">Instruction-tuned</span>
            <span class="tag">Қазақша</span>
        </div>
    </div>
    """)

    model_selector = gr.Radio(
        choices=list(MODELS.keys()),
        value=DEFAULT_MODEL,
        label="Модельді таңдаңыз",
        interactive=True,
    )

    with gr.Column(elem_id="sozkz-chat"):
        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(
                height=None,
                show_label=False,
                layout="bubble",
                bubble_full_width=False,
                placeholder=None,
                show_share_button=False,
                show_copy_button=False,
            ),
            textbox=gr.Textbox(
                placeholder="Сұрағыңызды жазыңыз...",
                show_label=False,
                container=False,
                scale=7,
            ),
            additional_inputs=[model_selector],
            examples=EXAMPLES,
            cache_examples=False,
        )

    gr.HTML("""
    <div id="sozkz-footer">
        <a href="https://huggingface.co/stukenov/sozkz-core-llama-600m-kk-instruct-v1">600M</a>&ensp;·&ensp;
        <a href="https://huggingface.co/stukenov/sozkz-core-llama-300m-kk-instruct-v1">300M</a>&ensp;·&ensp;
        <a href="https://huggingface.co/stukenov/sozkz-core-llama-150m-kk-instruct-v1">150M</a>&ensp;·&ensp;
        <a href="https://huggingface.co/stukenov/sozkz-core-llama-1b-kk-base-v1">Base 1B</a>&ensp;·&ensp;
        <a href="https://huggingface.co/spaces/stukenov/sozkz-kazakh-asr-demo">ASR</a>&ensp;·&ensp;
        <a href="https://huggingface.co/stukenov">stukenov</a>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
