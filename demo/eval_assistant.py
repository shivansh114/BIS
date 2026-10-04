"""Evaluation with HAND-WRITTEN questions (phrased like real users, not copied from catalogue names).

Answer key: eval_questions.csv (reviewers mark the last 2 columns). A question is correct if the returned IS number belongs to a product whose name
contains one of the listed key phrases. Team should double-check this key before quoting numbers.

  python eval_assistant.py          (retrieval + rules, no model needed)
"""
import statistics, time
import app_local as A

import csv as _csv
PRODUCT_QS = [(r["question"], [k.strip().lower() for k in r["key_phrases"].split(";")])
              for r in _csv.DictReader(open(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 and __import__("sys").argv[1].endswith(".csv") else "eval_questions.csv", encoding="utf-8"))]
OFF_TOPIC = ["how to cook biryani", "cricket score today", "write me a poem", "bitcoin price", "weather in jaipur",
             "train timings delhi", "who is the prime minister", "best phone under 10000", "movie tickets",
             "pizza near me", "flying car", "invisible cloak", "time machine parts", "love letter", "netflix password",
             "homework help maths", "share price of cement company", "how to lose weight", "tell me a joke", "song lyrics"]
BROAD = ["cement", "valve", "cable", "pipe", "steel", "wire", "battery", "lamp", "paint", "glass",
         "helmet", "cooker", "fan", "tank", "water"]


def decide(q):
    c = A.get_candidates(q); o = A.get_disambiguation_options(q, c)
    return c, o, bool(c) and c[0][1] >= A.CONFIDENCE_THRESHOLD


def right(prod_is, keys):
    names = [p["product_name"].lower() for p in A._products + A.CATALOG if p["is_standard"] == prod_is]
    return any(k in n for n in names for k in keys)


ok = clar = wrong = none = 0; misses = []; t = []
for q, keys in PRODUCT_QS:
    t0 = time.perf_counter(); c, o, conf = decide(q); t.append((time.perf_counter() - t0) * 1000)
    if o and any(right(x["is_standard"], keys) for x in o): clar += 1
    elif conf and not o and right(c[0][0]["is_standard"], keys): ok += 1
    elif not conf and not o: none += 1; misses.append((q, "no answer"))
    else: wrong += 1; misses.append((q, (o and "clarify w/o right one") or c[0][0]["product_name"]))
n = len(PRODUCT_QS)
off = sum(1 for q in OFF_TOPIC if not decide(q)[2] and not decide(q)[1])
br = sum(1 for q in BROAD if decide(q)[1])
print(f"Hand-written product questions: {n}")
print(f"  right standard: {ok + clar}/{n} ({round(100*(ok+clar)/n)}%)  [direct {ok}, via clarifying question {clar}]  wrong {wrong}, no answer {none}")
for q, why in misses: print(f"    MISS: {q!r} -> {why}")
print(f"Off-topic correctly declined: {off}/{len(OFF_TOPIC)}")
print(f"Broad words -> clarifying question: {br}/{len(BROAD)}")
print(f"Search time: median {statistics.median(t):.1f} ms")
