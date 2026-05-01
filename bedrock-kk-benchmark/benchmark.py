#!/usr/bin/env python3
"""
Benchmark Amazon Bedrock models for Kazakh language support.

Sends 20 Kazakh test questions to each available text model,
scores responses with Claude Sonnet 4.6 as judge, produces a final ranking.
Runs models in parallel for speed.

Usage:
    python benchmark.py                    # full benchmark (10 parallel)
    python benchmark.py --workers 20       # 20 models in parallel
    python benchmark.py --list             # list models only
    python benchmark.py --resume           # resume interrupted run
    python benchmark.py --models MODEL...  # test specific models
    python benchmark.py --ranking          # rebuild ranking from saved data
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import boto3

REGION = "us-east-1"
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-6"
OUT_DIR = Path(__file__).parent / "results"
RESPONSES_FILE = OUT_DIR / "responses.jsonl"
RANKING_FILE = OUT_DIR / "ranking.json"

FILE_LOCK = threading.Lock()
PROFILE_LOCK = threading.Lock()
PROFILE_CACHE = {}
COMPLETED = {}
COMPLETED_LOCK = threading.Lock()

# ── 20 test questions ────────────────────────────────────────────────────────

QUESTIONS = [
    {
        "id": 1,
        "question": "Сәлем! Сенің атың кім? Өзің туралы қазақша айтып бер.",
        "category": "conversation",
        "criteria": "Should greet back and respond in Kazakh about itself",
    },
    {
        "id": 2,
        "question": "Қазақстанның астанасы қай қала?",
        "category": "factual",
        "criteria": "Should answer 'Астана' in Kazakh",
    },
    {
        "id": 3,
        "question": "Мына сөйлемді орысшаға аудар: 'Бүгін ауа райы өте жақсы, серуенге шығайық'",
        "category": "translation_kk_ru",
        "criteria": "Should translate to Russian: approx 'Сегодня погода очень хорошая, давайте пойдём на прогулку'",
    },
    {
        "id": 4,
        "question": "Мына сөйлемді ағылшынша аудар: 'Мен қазақ тілін үйреніп жатырмын, бұл өте қызықты'",
        "category": "translation_kk_en",
        "criteria": "Should translate to English: 'I am learning the Kazakh language, it is very interesting'",
    },
    {
        "id": 5,
        "question": "'Отан — оттан да ыстық' мақалының мағынасын түсіндір.",
        "category": "proverb",
        "criteria": "Should explain the proverb about homeland being dearer than fire/warmth, ideally in Kazakh",
    },
    {
        "id": 6,
        "question": "Абай Құнанбаев кім болған? Оның қазақ әдебиетіне қосқан үлесі туралы айтып бер.",
        "category": "literature",
        "criteria": "Should describe Abai as a great Kazakh poet/philosopher, mention Qara Sözder or poems, in Kazakh",
    },
    {
        "id": 7,
        "question": "Есеп шығар: Айгүлдің 12 алмасы бар. Ол інісіне 5 алма берді, ал анасы оған тағы 3 алма берді. Айгүлде қанша алма қалды?",
        "category": "math",
        "criteria": "Should solve: 12 - 5 + 3 = 10 apples, explain in Kazakh",
    },
    {
        "id": 8,
        "question": "Қазақ тіліндегі дауысты дыбыстарды ата және жіңішке мен жуан дауыстыларға бөл.",
        "category": "linguistics",
        "criteria": "Should list Kazakh vowels and classify as front (ә,е,і,ө,ү) and back (а,о,ұ,ы)",
    },
    {
        "id": 9,
        "question": "Қазақстан туралы 5 сөйлем жаз.",
        "category": "generation",
        "criteria": "Should write 5 grammatically correct sentences about Kazakhstan in Kazakh",
    },
    {
        "id": 10,
        "question": "'Жақсы' сөзінің антонимін және 'үлкен' сөзінің синонимін айт.",
        "category": "vocabulary",
        "criteria": "Antonym of жақсы → жаман; synonym of үлкен → зор/ірі",
    },
    {
        "id": 11,
        "question": "Қазақ халқының дәстүрлі ұлттық ойындарын кемінде 5 ата.",
        "category": "culture",
        "criteria": "Should name 5+ traditional games: көкпар, бәйге, қыз қуу, тоғызқұмалақ, алтыбақан, аударыспақ, etc.",
    },
    {
        "id": 12,
        "question": (
            "Мына мәтінді қысқаша қорытындыла: 'Қазақстан — Орталық Азиядағы ең үлкен мемлекет. "
            "Аумағы 2,7 миллион шаршы километр. Халық саны 20 миллионнан асады. "
            "Мемлекеттік тілі — қазақ тілі. Елдің байлығы — мұнай, газ, уран және басқа пайдалы қазбалар.'"
        ),
        "category": "summarization",
        "criteria": "Should provide a concise summary in Kazakh capturing the main points",
    },
    {
        "id": 13,
        "question": "Төмендегі сөйлемдердегі қателерді тап және дұрыстап жаз: 'Мен кеше мектепке бардым. Ол менін достым. Біз ертен кинога барамыс.'",
        "category": "grammar",
        "criteria": "Should find errors: менін→менің, барамыс→барамыз. First sentence is correct.",
    },
    {
        "id": 14,
        "question": "Қазақша 1-ден 10-ға дейін сана.",
        "category": "counting",
        "criteria": "Should count: бір, екі, үш, төрт, бес, алты, жеті, сегіз, тоғыз, он",
    },
    {
        "id": 15,
        "question": "'Көктем' туралы 4 жолдық қысқа өлең жаз.",
        "category": "creative",
        "criteria": "Should write a short 4-line poem about spring in Kazakh",
    },
    {
        "id": 16,
        "question": "Қазақ тіліндегі септік жалғауларын атап, әрқайсысына мысал келтір.",
        "category": "grammar_deep",
        "criteria": "Should list 7 Kazakh cases (атау, ілік, барыс, табыс, жатыс, шығыс, көмектес) with examples",
    },
    {
        "id": 17,
        "question": "Алматы мен Астананың айырмашылықтарын салыстыр. Қайсысы туризм үшін жақсы?",
        "category": "comparison",
        "criteria": "Should compare Almaty and Astana (climate, geography, attractions) in Kazakh",
    },
    {
        "id": 18,
        "question": "Қазақ тойының дәстүрлері мен рәсімдерін сипатта. Қыз ұзату той қалай өтеді?",
        "category": "culture_deep",
        "criteria": "Should describe Kazakh wedding traditions (құда түсу, қыз ұзату, нике қию) in Kazakh",
    },
    {
        "id": 19,
        "question": "Келесі әңгімені жалғастыр (кемінде 5 сөйлем): 'Бір бала таудан тас тауып алды. Ол тас түнде жарқырап жанатын еді...'",
        "category": "story",
        "criteria": "Should continue the story creatively in Kazakh with at least 5 sentences",
    },
    {
        "id": 20,
        "question": (
            "Келесі мәтінді ағылшыннан қазақшаға аудар: 'Artificial intelligence is transforming education. "
            "Students can now learn at their own pace with personalized tutoring systems. "
            "However, human teachers remain essential for developing critical thinking and social skills.'"
        ),
        "category": "translation_en_kk",
        "criteria": "Should translate to grammatically correct Kazakh about AI in education",
    },
]

# ── Model filtering ──────────────────────────────────────────────────────────

SKIP_PATTERNS = [
    "embed", "titan-image", "stable-", "nova-canvas", "nova-reel",
    "nova-sonic", "rerank", "twelvelabs", "safeguard", "upscale",
    "outpaint", "inpaint", "style-", "control-", "search-", "remove-background",
    "erase-object", "pegasus",
]


def should_skip(model_id: str) -> bool:
    low = model_id.lower()
    return any(p in low for p in SKIP_PATTERNS)


def is_context_variant(model_id: str) -> bool:
    parts = model_id.split(":")
    if len(parts) >= 3:
        tail = parts[-1]
        if tail.endswith("k") and tail[:-1].isdigit():
            return True
        if tail == "mm":
            return True
    return False


def get_text_models(bedrock):
    resp = bedrock.list_foundation_models()
    models = []
    seen = set()
    for m in resp["modelSummaries"]:
        mid = m["modelId"]
        if should_skip(mid) or is_context_variant(mid) or mid in seen:
            continue
        if "TEXT" not in m.get("outputModalities", []):
            continue
        seen.add(mid)
        models.append({
            "modelId": mid,
            "modelName": m.get("modelName", mid),
            "provider": m.get("providerName", "Unknown"),
        })
    PRIORITY = [
        "qwen.", "deepseek.", "anthropic.", "mistral.mistral-large",
        "openai.", "zai.", "meta.llama4", "meta.llama3-3", "meta.llama3-2-90b",
        "meta.llama3-1-70b", "meta.llama3-70b", "google.gemma-3-27b",
        "mistral.", "moonshot", "meta.", "google.", "nvidia.",
        "minimax.", "cohere.", "amazon.", "ai21.", "writer.",
    ]

    def priority_key(m):
        mid = m["modelId"]
        for i, prefix in enumerate(PRIORITY):
            if mid.startswith(prefix):
                return i
        return len(PRIORITY)

    return sorted(models, key=priority_key)


# ── API calls ────────────────────────────────────────────────────────────────

def call_model(runtime, model_id, text, max_tokens=500, temperature=0.1, retries=2):
    with PROFILE_LOCK:
        mid = PROFILE_CACHE.get(model_id, model_id)
    for attempt in range(retries + 1):
        try:
            resp = runtime.converse(
                modelId=mid,
                messages=[{"role": "user", "content": [{"text": text}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            return {
                "text": resp["output"]["message"]["content"][0]["text"],
                "input_tokens": resp["usage"]["inputTokens"],
                "output_tokens": resp["usage"]["outputTokens"],
                "error": None,
            }
        except Exception as e:
            err_str = str(e)
            name = type(e).__name__
            if "Throttl" in name and attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            if "inference profile" in err_str.lower() and mid == model_id:
                mid = f"us.{model_id}"
                with PROFILE_LOCK:
                    PROFILE_CACHE[model_id] = mid
                continue
            return {"text": None, "input_tokens": 0, "output_tokens": 0,
                    "error": f"{name}: {err_str[:300]}"}


def judge(runtime, question, criteria, response_text):
    if not response_text:
        return {"understanding": 0, "kazakh_quality": 0, "correctness": 0, "comment": "No response"}

    prompt = f"""You are a Kazakh language expert evaluating an AI model's response to a Kazakh question.

QUESTION (Kazakh): {question}
EXPECTED: {criteria}
MODEL RESPONSE: {response_text}

Rate on three axes (0–10 each):
1. understanding — Did the model understand the Kazakh question? 0 = total failure, 10 = perfect.
2. kazakh_quality — Quality of Kazakh in the response? 0 = no Kazakh / gibberish, 5 = understandable but errors, 10 = native-level.
   If the question asks for translation OUT of Kazakh (to Russian/English), score the target language quality instead.
3. correctness — Is the answer factually correct and meets the criteria? 0 = wrong, 10 = perfect.

Return ONLY valid JSON (no markdown, no extra text):
{{"understanding": N, "kazakh_quality": N, "correctness": N, "comment": "brief English comment"}}"""

    result = call_model(runtime, JUDGE_MODEL, prompt, max_tokens=200, temperature=0.0)
    if result["error"] or not result["text"]:
        return {"understanding": -1, "kazakh_quality": -1, "correctness": -1,
                "comment": f"Judge error: {result['error']}"}
    try:
        txt = result["text"].strip()
        if "```" in txt:
            txt = txt.split("```")[1]
            if txt.startswith("json"):
                txt = txt[4:]
            txt = txt.strip()
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"understanding": -1, "kazakh_quality": -1, "correctness": -1,
                "comment": f"Judge returned invalid JSON: {result['text'][:100]}"}


# ── File I/O ─────────────────────────────────────────────────────────────────

def save_entry(entry):
    with FILE_LOCK:
        with open(RESPONSES_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with COMPLETED_LOCK:
        COMPLETED[(entry["model_id"], entry["question_id"])] = entry


def load_completed():
    done = {}
    if RESPONSES_FILE.exists():
        with open(RESPONSES_FILE) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    done[(entry["model_id"], entry["question_id"])] = entry
    return done


# ── Per-model worker ─────────────────────────────────────────────────────────

def benchmark_model(model, skip_judge=False):
    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    mid = model["modelId"]
    prov = model["provider"]
    results = []
    consecutive_errors = 0

    for q in QUESTIONS:
        key = (mid, q["id"])
        with COMPLETED_LOCK:
            if key in COMPLETED:
                continue

        result = call_model(runtime, mid, q["question"])

        scores = {"understanding": 0, "kazakh_quality": 0, "correctness": 0, "comment": ""}
        if not skip_judge and result["text"]:
            scores = judge(runtime, q["question"], q["criteria"], result["text"])

        entry = {
            "model_id": mid,
            "provider": prov,
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "response": result["text"],
            "error": result["error"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "scores": scores,
            "ts": datetime.now().isoformat(),
        }
        save_entry(entry)

        if result["error"]:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                for rq in QUESTIONS:
                    rkey = (mid, rq["id"])
                    with COMPLETED_LOCK:
                        if rkey in COMPLETED:
                            continue
                    skip_entry = {
                        "model_id": mid, "provider": prov,
                        "question_id": rq["id"], "category": rq["category"],
                        "question": rq["question"], "response": None,
                        "error": "skipped_after_3_errors",
                        "input_tokens": 0, "output_tokens": 0,
                        "scores": {"understanding": 0, "kazakh_quality": 0, "correctness": 0, "comment": "Skipped"},
                        "ts": datetime.now().isoformat(),
                    }
                    save_entry(skip_entry)
                return mid, "SKIPPED (3 errors)"
        else:
            consecutive_errors = 0

    with COMPLETED_LOCK:
        total = 0
        answered = 0
        for q in QUESTIONS:
            e = COMPLETED.get((mid, q["id"]))
            if e and not e["error"]:
                answered += 1
                sc = e["scores"]
                total += max(0, sc.get("understanding", 0)) + max(0, sc.get("kazakh_quality", 0)) + max(0, sc.get("correctness", 0))
    return mid, f"{total}/{answered*30} ({total/(answered*30)*100:.0f}%)" if answered > 0 else "no answers"


# ── Ranking ──────────────────────────────────────────────────────────────────

def build_ranking(completed):
    by_model = {}
    for entry in completed.values():
        mid = entry["model_id"]
        if mid not in by_model:
            by_model[mid] = {
                "model_id": mid,
                "provider": entry["provider"],
                "answered": 0, "errored": 0,
                "understanding": 0, "kazakh_quality": 0, "correctness": 0,
                "total": 0,
                "input_tokens": 0, "output_tokens": 0,
            }
        s = by_model[mid]
        if entry["error"]:
            s["errored"] += 1
        else:
            s["answered"] += 1
            sc = entry.get("scores", {})
            u = max(0, sc.get("understanding", 0))
            k = max(0, sc.get("kazakh_quality", 0))
            c = max(0, sc.get("correctness", 0))
            s["understanding"] += u
            s["kazakh_quality"] += k
            s["correctness"] += c
            s["total"] += u + k + c
            s["input_tokens"] += entry.get("input_tokens", 0)
            s["output_tokens"] += entry.get("output_tokens", 0)

    ranking = sorted(by_model.values(), key=lambda x: x["total"], reverse=True)
    for r in ranking:
        mx = r["answered"] * 30
        r["score_pct"] = round(r["total"] / mx * 100, 1) if mx > 0 else 0.0
    return ranking


def print_ranking(ranking):
    print(f"\n{'='*100}")
    print(f"  KAZAKH LANGUAGE SUPPORT RANKING — Amazon Bedrock")
    print(f"{'='*100}")
    print(f"{'#':<4} {'Provider':<16} {'Model':<44} {'Score':>8} {'%':>7}  {'Und':>4} {'KzQ':>4} {'Cor':>4}  {'Q':>3} {'Err':>3}")
    print("-" * 100)
    for i, r in enumerate(ranking, 1):
        mx = r["answered"] * 30
        score_str = f"{r['total']}/{mx}" if mx > 0 else "—"
        print(
            f"{i:<4} {r['provider']:<16} {r['model_id']:<44} {score_str:>8} {r['score_pct']:>6.1f}%"
            f"  {r['understanding']:>4} {r['kazakh_quality']:>4} {r['correctness']:>4}"
            f"  {r['answered']:>3} {r['errored']:>3}"
        )
    print(f"\nUnd=Understanding  KzQ=Kazakh Quality  Cor=Correctness  Q=Questions answered  Err=Errors")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Bedrock models for Kazakh language")
    parser.add_argument("--list", action="store_true", help="List available text models")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    parser.add_argument("--ranking", action="store_true", help="Rebuild ranking from saved data")
    parser.add_argument("--models", nargs="*", help="Test only these model IDs")
    parser.add_argument("--skip-judge", action="store_true", help="Collect responses without scoring")
    parser.add_argument("--workers", type=int, default=10, help="Parallel model workers (default 10)")
    args = parser.parse_args()

    bedrock = boto3.client("bedrock", region_name=REGION)

    if args.list:
        models = get_text_models(bedrock)
        print(f"\nAvailable text models ({len(models)}):\n")
        for m in models:
            print(f"  {m['provider']:<16} {m['modelId']}")
        return

    if args.ranking:
        completed = load_completed()
        if not completed:
            print("No results found. Run the benchmark first.")
            return
        ranking = build_ranking(completed)
        print_ranking(ranking)
        with open(RANKING_FILE, "w") as f:
            json.dump(ranking, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {RANKING_FILE}")
        return

    # ── Full benchmark
    models = get_text_models(bedrock)
    if args.models:
        models = [m for m in models if m["modelId"] in args.models]
        if not models:
            print("None of the specified models found. Use --list to see available models.")
            return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global COMPLETED
    if args.resume:
        COMPLETED = load_completed()

    print(f"\n{'='*70}")
    print(f"  Kazakh Language Benchmark — Amazon Bedrock")
    print(f"  Models: {len(models)}  |  Questions: {len(QUESTIONS)}  |  Workers: {args.workers}")
    print(f"  Judge: {JUDGE_MODEL}")
    if COMPLETED:
        print(f"  Resuming: {len(COMPLETED)} responses cached")
    print(f"{'='*70}\n")

    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(benchmark_model, model, args.skip_judge): model
            for model in models
        }
        for future in as_completed(futures):
            model = futures[future]
            done_count += 1
            try:
                mid, status = future.result()
                print(f"  [{done_count}/{len(models)}] {model['provider']:<16} {mid:<44} → {status}")
            except Exception as e:
                print(f"  [{done_count}/{len(models)}] {model['provider']:<16} {model['modelId']:<44} → CRASH: {e}")

    # ── Final ranking
    with COMPLETED_LOCK:
        ranking = build_ranking(COMPLETED)
    print_ranking(ranking)

    with open(RANKING_FILE, "w") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)
    print(f"\nResults: {RESPONSES_FILE}")
    print(f"Ranking: {RANKING_FILE}")


if __name__ == "__main__":
    main()
