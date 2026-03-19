import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "stukenov/sozkz-core-llama-300m-kk-gec-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.eval()

def correct(tag, text, max_tokens=150):
    prompt = f"<{tag}> {text}\n\u2192 "
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.3, top_p=0.9, do_sample=True,
            repetition_penalty=1.1,
        )
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract after arrow
    if "\u2192 " in result:
        result = result.split("\u2192 ", 1)[1]
    # Stop at newline or next tag
    for stop in ["\n<", "\n\n"]:
        if stop in result:
            result = result[:result.index(stop)]
    return result.strip()

# Test cases: each has 1-2 errors per sentence
tests = [
    # 1. Сингармонизм (vowel_harmony) — жуан/жіңішке дауыстылар
    ("сингармонизм", "Қабырғаларде ұлттық өрнектер мен суреттер бейнеленген.", "Қабырғаларда (де→да)"),
    ("сингармонизм", "Мектептің кітапханасінда көптеген кітаптар бар.", "кітапханасында (сінда→сында)"),

    # 2. Септік / Case suffix
    ("септік", "Ол өз жека басының пайдасына жинап алынатын салық түрін енгізді.", "жеке (жека→жеке)"),
    ("септік", "Балалар мектепке баратын жолда ойнайдылар.", "баратын жолда (correct or fix case)"),

    # 3. Жіктік / Personal ending
    ("жіктік", "Біз бүгін мектепке барамыз және сабақ оқимыз.", "оқимыз (оқимыз→оқимыз or оқыймыз)"),
    ("жіктік", "Тағамдар кеңінен қолданыламыз.", "қолданылады (ламыз→лады)"),

    # 4. Болымсыз / Negation
    ("болымсыз", "Бұл кітап қызықты емес кітап.", "extra емес"),
    ("болымсыз", "Ол бүгін мектепке бармады емес.", "double negation"),

    # 5. Шақ / Tense
    ("шақ", "Бұл топқа көптеген нәруыздар жатаады.", "жатады (жатаады→жатады)"),
    ("шақ", "Кеше мен кітап оқыймын.", "оқыдым (оқыймын→оқыдым, past tense)"),

    # 6. Сөз тәртібі / Word order
    ("сөз_тәртібі", "Базарға мен кеше бардым жемістер алуға.", "Мен кеше базарға жемістер алуға бардым."),
    ("сөз_тәртібі", "Жақсы оқиды ол мектепте.", "Ол мектепте жақсы оқиды."),

    # 7. Құрмалас / Complex sentence
    ("құрмалас", "ауытқуы бар болуы мүмкін. немесе Ол тәсілдер көп таралған.", "reorder clauses"),
    ("құрмалас", "Келді. Және сөйледі ол кеше.", "join into proper sentence"),

    # 8. Тәуелдік / Possessive
    ("тәуелдік", "Оның түпкы мәніне сай болмағандықтан ол жаңа жол ұсынды.", "түпкі (түпкы→түпкі)"),
    ("тәуелдік", "Менің анамнің үйі ауылда орналасқан.", "анамның (анамнің→анамның)"),

    # 9. Шылау / Postposition
    ("шылау", "Тіл турелі толғаныс жазды.", "туралы (турелі→туралы)"),
    ("шылау", "Ол мектеп ушін көп жұмыс істеді.", "үшін (ушін→үшін)"),

    # 10. Көптік / Plural
    ("көптік", "Олар жарылғанлер магмадан түзілген.", "жарылған (лер extra)"),
    ("көптік", "Студенттар университетте оқиды.", "Студенттер (тар→тер)"),

    # 11. Жалғау / General case suffix
    ("жалғау", "Барлық тірі организмдер біртекті құрылымын көрсетті.", "организмдердің (case suffix)"),
    ("жалғау", "Мен достарым бірге саяхатқа шықтым.", "достарыммен (missing бірге suffix)"),

    # 12. Қате / Noise/typo
    ("қате", "Әулие Эндрюдін үш кресті батылдық, беріктік жане мейірімділік білдіреді.", "Эндрюдің, және (typos)"),
    ("қате", "Қазақстнаның астанасы Астна қаласы.", "Қазақстанның, Астана (typos)"),

    # 13. Грамматика / General
    ("грамматика", "Олар шын пейілдерімен боліседі бір-біріне.", "бөліседі, бір-бірімен"),
    ("грамматика", "Мен кеше дүкенге барды жаңа кітап алды.", "бардым, алдым (person agreement)"),

    # 14. Таза / Clean (should return unchanged)
    ("таза", "Қазақстан Орталық Азиядағы ең ірі мемлекет.", "should be unchanged"),
    ("таза", "Абай Құнанбайұлы ұлы қазақ ақыны.", "should be unchanged"),
]

print("=" * 90)
print(f"GEC MODEL TEST: {model_id}")
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print("=" * 90)

current_tag = ""
for tag, input_text, expected in tests:
    if tag != current_tag:
        current_tag = tag
        print(f"\n{'='*90}")
        print(f"  <{tag}>")
        print(f"{'='*90}")

    output = correct(tag, input_text)
    # Check if output differs from input
    changed = "CHANGED" if output != input_text else "SAME"

    print(f"\n  INPUT:    {input_text}")
    print(f"  OUTPUT:   {output}")
    print(f"  EXPECTED: {expected}")
    print(f"  STATUS:   {changed}")
    print(f"  {'-'*86}")

print(f"\n{'='*90}")
print("DONE")
