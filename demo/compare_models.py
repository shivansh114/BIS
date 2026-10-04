"""Does fine-tuning help?  Base Llama 3.1 8B vs our fine-tuned bis-assistant, same questions, same retrieved data.

    python compare_models.py            (needs Ollama running with both models: ollama list)

Writes compare_results.csv. Auto-checks per answer:
  right_is     - the expected IS number appears in the answer
  made_up_is   - the answer quotes an IS number that was NOT in the retrieved data  (hallucination signal)
  language_ok  - reply script matches the question (Devanagari vs Roman)
Then 2 teammates fill reviewer columns (correct / partly / wrong) in the CSV.
"""
import csv, re, statistics, time
import ollama
import app_local as A

MODELS = ["llama3.1:8b", "bis-assistant"]
IS_RX = re.compile(r"IS\s*/?\s*(?:IEC\s*)?\d{2,5}", re.I)
norm = lambda x: re.sub(r"\s+", " ", x.upper().replace("/", " ")).strip()

qs = [r["question"] for r in csv.DictReader(open("eval_questions.csv", encoding="utf-8"))]
qs += ["toothpaste ka standard kya hai", "हेलमेट के लिए कौन सा मानक है", "LED bulb ka ISI chahiye"]
rows, summary = [], {m: {"right": 0, "made_up": 0, "lang": 0, "t": [], "n": 0} for m in MODELS}
for q in qs:
    cands = A.get_candidates(q); opts = A.get_disambiguation_options(q, cands)
    if opts or not cands or cands[0][1] < A.CONFIDENCE_THRESHOLD:
        continue                                     # compare only where retrieval gave the model real data
    ctx = A.build_context_block(cands, A._workflow_steps)
    allowed = {norm(m) for m in IS_RX.findall(ctx)}
    expected = norm(IS_RX.findall(cands[0][0]["is_standard"])[0])
    prompt = f"RETRIEVED DATA:\n{ctx}\n\n{A.build_language_directive(q)}\n\nUSER MESSAGE:\n{q}\n\nRespond following the system rules."
    for m in MODELS:
        t0 = time.perf_counter()
        ans = ollama.chat(model=m, messages=[{"role": "system", "content": A.SYSTEM_PROMPT},
                                             {"role": "user", "content": prompt}], options={"temperature": 0.2})["message"]["content"]
        dt = time.perf_counter() - t0
        found = {norm(x) for x in IS_RX.findall(ans)}
        right, made_up = expected in found, bool(found - allowed)
        lang_ok = bool(A._DEVANAGARI_PATTERN.search(ans)) == bool(A._DEVANAGARI_PATTERN.search(q))
        S = summary[m]; S["n"] += 1; S["right"] += right; S["made_up"] += made_up; S["lang"] += lang_ok; S["t"].append(dt)
        rows.append({"question": q, "model": m, "answer": ans, "right_is": right, "made_up_is": made_up,
                     "language_ok": lang_ok, "seconds": round(dt, 1), "reviewer_1": "", "reviewer_2": ""})
        print(f"{m:15} {'OK ' if right else 'MISS'} {'MADE-UP' if made_up else '       '} {dt:5.1f}s  {q[:45]}")

with open("compare_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\n=== put this table on slide 6 ===")
print(f"{'model':15} {'right IS':>9} {'made-up IS':>11} {'right language':>15} {'median time':>12}")
for m, S in summary.items():
    n = max(S["n"], 1)
    print(f"{m:15} {100*S['right']//n:>8}% {100*S['made_up']//n:>10}% {100*S['lang']//n:>14}% {statistics.median(S['t']) if S['t'] else 0:>10.1f}s")
print(f"questions compared: {summary[MODELS[0]]['n']}   ->  now 2 teammates fill reviewer_1 / reviewer_2 in compare_results.csv")
