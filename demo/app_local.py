"""
BIS AI Assistant - Local demo (our fine-tuned model via Ollama)
----------------------------------------------------------------
Model : bis-assistant  (Llama 3.1 8B + our LoRA, exported to GGUF, see Modelfile)
Data  : bis_full_dataset.json  (865 products, 6 schemes)

Setup (once):
    ollama create bis-assistant -f Modelfile
    pip install flask ollama scikit-learn reportlab opencv-python-headless

Run:
    python app_local.py      ->  http://127.0.0.1:5000
                                 http://127.0.0.1:5000/dashboard  (analytics)

If Ollama isn't running the app still answers (plain template replies),
so the demo never dies in front of the judges.
"""

import csv
import datetime
import io
import json
import math
import os
import re as _re
import sqlite3
import sys

from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama
except ImportError:
    ollama = None
    print("ollama package missing (pip install ollama) - running in offline template mode")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "bis_full_dataset.json")
DB_FILE = os.path.join(HERE, "analytics.db")
MODEL_NAME = os.environ.get("BIS_MODEL", "bis-assistant")   # our fine-tuned model
LANGUAGE_MODE = "auto"

app = Flask(__name__, static_folder="static")


# ---------- Data loading & retrieval ----------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    _data = json.load(f)

_products = _data["products"]
_workflow_steps = _data["certification_workflow_steps"]

_corpus = [f"{p['product_name']} {p['category']} {p['scope_summary']}" for p in _products]
def _tok(text):
    # crude plural handling so "cooker" matches "Cookers"
    out = []
    for w in _re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 4 and w.endswith("es") and not w.endswith("ses"):
            w = w[:-2] if w[-3] in "hx" else w[:-1]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


_vectorizer = TfidfVectorizer(tokenizer=_tok, token_pattern=None, stop_words="english", ngram_range=(1, 2))
_tfidf_matrix = _vectorizer.fit_transform(_corpus)
_cat_vec = TfidfVectorizer(tokenizer=lambda t: _tok(t), token_pattern=None, stop_words="english", ngram_range=(1, 2)) if False else None
# second vectorizer on character chunks so typos ("presure cookar") still match
_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
_char_matrix = _char_vec.fit_transform([p["product_name"].lower() for p in _products])

# ---------- full BIS catalogue (23,813 standards) ----------
# set BIS_DATA_DIR to the folder with standards_enriched.csv
CATALOG, CATALOG_TEXT = [], []
_cat_path = os.path.join(os.environ.get("BIS_DATA_DIR", HERE), "standards_enriched.csv")
if os.path.exists(_cat_path):
    with open(_cat_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("withdrawn_flag", "0")).strip() in ("1", "True"):
                continue
            cls = (r.get("certification_classification") or "").strip()
            CATALOG.append({"product_name": r["title"].split("—")[0].strip(), "category": r.get("group", ""),
                            "is_standard": r["standard_number"].split(":")[0].strip(),
                            "scheme": cls or "not stated", "mandatory": cls.lower().startswith(("mandatory", "compulsory")),
                            "legal_basis": "BIS catalogue: " + (cls or "classification not stated"),
                            "scope_summary": r["title"], "regulating_ministry": r.get("ministry", "BIS"),
                            "source_url": r.get("source_url", "")})
            CATALOG_TEXT.append(f"{r['title']} {r.get('subgroup', '')} {r.get('subsubgroup', '')}")
    print(f"Full catalogue loaded: {len(CATALOG)} current standards")

if CATALOG:
    _cat_vec = TfidfVectorizer(tokenizer=_tok, token_pattern=None, stop_words="english", ngram_range=(1, 2))
    _cat_matrix = _cat_vec.fit_transform(CATALOG_TEXT)

# ---------- semantic search: multilingual embeddings (falls back to TF-IDF if not installed) ----------
EMBED_MODEL = os.environ.get("BIS_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ALL_ITEMS = _products + CATALOG
ALL_TEXT = [f"{p['product_name']}. {p['scope_summary']}" for p in _products] + CATALOG_TEXT
_emb_model, _emb_matrix = None, None
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _emb_model = SentenceTransformer(EMBED_MODEL)        # downloads once, then works offline
    _cache = os.path.join(HERE, f"emb_cache_{len(ALL_TEXT)}.npy")
    if os.path.exists(_cache):
        _emb_matrix = np.load(_cache)
    else:
        print(f"Embedding {len(ALL_TEXT)} standards once (a few minutes on CPU)...")
        _emb_matrix = _emb_model.encode(ALL_TEXT, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        np.save(_cache, _emb_matrix)
    _all_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    _all_char = _all_char_vec.fit_transform([t.lower() for t in ALL_TEXT])
    _all_word_vec = TfidfVectorizer(tokenizer=_tok, token_pattern=None, stop_words="english", ngram_range=(1, 2))
    _all_word = _all_word_vec.fit_transform(ALL_TEXT)
    _all_title_tokens = [set(_tok(p["product_name"])) for p in ALL_ITEMS]
    print(f"Semantic search ON ({EMBED_MODEL})")
except Exception as e:
    _emb_model = None
    print(f"Semantic search OFF, using TF-IDF ({e.__class__.__name__}: pip install sentence-transformers)")


def semantic_candidates(q, top_n=15):
    qv = _emb_model.encode([q], normalize_embeddings=True)[0]
    sem = _emb_matrix @ qv
    chars = cosine_similarity(_all_char_vec.transform([q]), _all_char).flatten()
    words = cosine_similarity(_all_word_vec.transform([q]), _all_word).flatten()
    score = 0.5 * sem + 0.3 * words + 0.1 * chars
    # title coverage: how many of the user's content words are in the standard's title
    qt = {w for w in _tok(q) if len(w) > 2} - set(_all_word_vec.get_stop_words())
    if qt:
        top_pool = score.argsort()[::-1][:60]
        for i in top_pool:
            score[i] += 0.15 * len(qt & _all_title_tokens[i]) / len(qt)
    score[:len(_products)] += 0.03          # small preference for the curated product records
    idx = score.argsort()[::-1][:top_n]
    return [(ALL_ITEMS[i], float(score[i])) for i in idx]


# Hindi words -> English, and Hinglish filler words that only confuse TF-IDF.
# (the original query still goes to the model, this is only for retrieval)
HI_TO_EN = {"कुकर": "pressure cooker", "प्रेशर": "pressure", "हेलमेट": "helmet", "सीमेंट": "cement",
            "सरिया": "steel bars", "बल्ब": "led lamp", "खिलौना": "toys", "खिलौने": "toys", "पानी": "water",
            "बोतल": "bottle", "सोना": "gold", "चांदी": "silver", "मोबाइल": "mobile phone", "पंखा": "fan",
            "तार": "cable", "प्रेस": "electric iron", "टूथपेस्ट": "toothpaste", "बैटरी": "battery"}
FILLER = {"ka", "ki", "ke", "ko", "kya", "hai", "batao", "mujhe", "chahiye", "liye", "kaise", "kaunsa",
          "konsa", "kaun", "milega", "baare", "mein", "standard", "scheme", "certification", "please", "bhai",
          "का", "की", "के", "क्या", "है", "मुझे", "चाहिए", "लिए", "कौन", "सा", "मानक", "कैसे", "मिलेगा"}


COMMON_NAMES = {   # how people say it -> how BIS titles say it (general vocabulary, ~60 terms)
    "house wiring": "pvc insulated cable wiring", "home wiring": "pvc insulated cable wiring", "wiring": "cable wiring",
    "grey cement": "portland cement", "gray cement": "portland cement", "ghar ka cement": "portland cement",
    "saria": "steel bars reinforcement", "sariya": "steel bars reinforcement", "tmt": "deformed steel bars reinforcement",
    "rebar": "deformed steel bars reinforcement", "bulb": "lamp", "tubelight": "fluorescent lamp", "tube light": "fluorescent lamp",
    "geyser": "water heater", "mixie": "mixer grinder", "mixer": "mixer grinder", "power bank": "portable power bank lithium",
    "charger": "adaptor charger power supply", "mobile": "mobile phone", "cellphone": "mobile phone", "laptop": "information technology equipment",
    "computer": "information technology equipment", "tv": "television", "fridge": "refrigerating appliance", "refrigerator": "refrigerating appliance",
    "ac": "air conditioner", "a.c.": "air conditioner", "cooler": "air cooler", "mcb": "miniature circuit breaker",
    "rccb": "residual current circuit breaker", "elcb": "residual current circuit breaker", "extension board": "socket outlet plug",
    "extension cord": "socket outlet plug", "kettle": "electric kettle water heating", "induction": "induction cooking hob",
    "cooktop": "cooking hob", "mineral water": "packaged drinking water", "bottled water": "packaged drinking water",
    "drinking water business": "packaged drinking water", "gas cylinder": "lpg cylinder", "wire": "cable", "almirah": "steel cupboard",
    "sanitary pads": "sanitary napkins", "pads": "sanitary napkins", "diaper": "baby diapers", "baby food": "infant food",
    "milk powder": "milk powder", "hallmark": "hallmarking gold jewellery fineness", "sona": "gold", "chandi": "silver",
    "helmet for bike": "helmet two wheeler riders", "bike helmet": "helmet two wheeler riders", "tyre": "pneumatic tyres",
    "tyres": "pneumatic tyres", "inverter battery": "lead acid storage battery", "battery for bike": "lead acid storage battery",
    "two wheeler battery": "lead acid storage battery", "pipes": "pipe", "nal": "water tap", "tap": "water tap",
    "safety shoes": "safety footwear", "chappal": "footwear", "juice bottle": "glass container", "plywood": "plywood",
    "sink": "kitchen sink", "chair": "chair", "cctv": "video surveillance", "pressure cooker": "domestic pressure cooker",
}


def retrieval_query(q):
    t = q.lower()
    for k, v in COMMON_NAMES.items():
        t = _re.sub(rf"\b{k}\b", v, t)
    for hi, en in HI_TO_EN.items():
        t = t.replace(hi, f" {en} ")
    words = [w for w in _re.findall(r"[\w\u0900-\u097F]+", t.lower()) if w not in FILLER]
    return spell_fix(" ".join(words)) or q


_VOCAB = None
_VOCAB_LIST = None


def _vocab():
    global _VOCAB, _VOCAB_LIST
    if _VOCAB is None:
        _VOCAB = set(w for t in _corpus + CATALOG_TEXT for w in _tok(t))
        _VOCAB_LIST = sorted(w for w in _VOCAB if len(w) > 3)
    return _VOCAB


def spell_fix(text):
    """'cabless for house wiring' -> 'cable for house wiring' (closest catalogue word, only if very close)"""
    import difflib
    v = _vocab(); out = []
    for w in text.split():
        t = _tok(w)
        if len(w) < 6 or not t or t[0] in v or not w.isalpha():   # short words: char-matching handles them
            out.append(w); continue
        m = [c for c in difflib.get_close_matches(w, _VOCAB_LIST, n=3, cutoff=0.83) if c[0] == w[0]]   # typos keep the first letter
        out.append(m[0] if m else w)
    return " ".join(out)


def in_scope(q):
    """'how to cook biryani' -> False. Most content words must appear in BIS titles/scopes."""
    _vocab()
    words = [w for w in _tok(q) if len(w) > 2 and w not in _vectorizer.get_stop_words()
             and w not in {"want", "make", "sell", "need", "standard", "product", "home", "house", "use", "used", "brand", "business",
                         "company", "manufacturing", "making", "kitchen", "bike", "door", "room", "children", "kid", "my"}]
    if not words:
        return False
    import difflib
    def known(w):   # exact catalogue word, or a near-miss typo of one ("cookr", "bulbb")
        return w in _VOCAB or (len(w) >= 5 and any(c[0] == w[0] for c in difflib.get_close_matches(w, _VOCAB_LIST, n=3, cutoff=0.85)))
    return sum(known(w) for w in words) / len(words) >= 0.6


def get_candidates(query, top_n=15, cutoff=0.05):
    q = retrieval_query(query)
    if not in_scope(q):
        return []
    if _emb_model is not None:
        return semantic_candidates(q, top_n)
    sims = cosine_similarity(_vectorizer.transform([q]), _tfidf_matrix).flatten()
    chars = cosine_similarity(_char_vec.transform([q]), _char_matrix).flatten()
    score = 0.7 * sims + 0.3 * chars
    ranked_idx = score.argsort()[::-1][:top_n]
    out = [(_products[i], float(score[i])) for i in ranked_idx if score[i] >= cutoff]
    if CATALOG and (not out or out[0][1] < CONFIDENCE_THRESHOLD):
        cs = cosine_similarity(_cat_vec.transform([q]), _cat_matrix).flatten()
        out = [(CATALOG[i], float(cs[i])) for i in cs.argsort()[::-1][:top_n] if cs[i] >= cutoff] or out
    return out


CONFIDENCE_THRESHOLD = 0.15  # below this, a "match" is likely noise, not a real product query


def get_disambiguation_options(query, candidates=None):
    candidates = candidates if candidates is not None else get_candidates(query)
    if not candidates:
        return None
    top_score = candidates[0][1]
    if top_score < CONFIDENCE_THRESHOLD:
        return None
    # every word the user typed is in the top product's name -> they meant that one
    # (fixes "pressure cooker" being shown Rice Cooker as an option)
    qwords = [w for w in retrieval_query(query).split() if len(w) > 2]
    name_words = candidates[0][0]["product_name"].lower().split()
    if len(qwords) >= 2 and all(any(n.startswith(w) or w.startswith(n) for n in name_words) for w in qwords):
        return None
    close_matches = [c for c in candidates if c[1] >= (top_score - AMBIG_GAP if _emb_model is not None else top_score * 0.75)]
    if len(retrieval_query(query).split()) == 1:          # "cement", "valve" - too broad, ask
        close_matches = [c for c in candidates if c[1] >= (top_score - 2 * AMBIG_GAP if _emb_model is not None else top_score * 0.5)]
    if len(close_matches) >= 3:
        return [{"product_name": p["product_name"], "is_standard": p["is_standard"]} for p, s in close_matches[:8]]
    return None


def build_context_block(candidates, workflow_steps, limit=3):
    blocks = []
    for product, score in candidates[:limit]:
        scheme = product["scheme"]
        steps = workflow_steps.get(scheme, [])
        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        mandatory_label = "Mandatory" if product["mandatory"] else "Voluntary"
        block = f"""
Product: {product['product_name']}
Category: {product['category']}
Applicable IS Standard: {product['is_standard']}
Certification Scheme: {scheme} ({mandatory_label})
Legal Basis: {product['legal_basis']}
Scope: {product['scope_summary']}
Regulating Body: {product['regulating_ministry']}
Certification Steps:
{steps_text if steps_text else "  (No mandatory workflow - voluntary certification only)"}
""".strip()
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


_DEVANAGARI_PATTERN = _re.compile(r'[\u0900-\u097F]')


def contains_devanagari(text):
    return bool(_DEVANAGARI_PATTERN.search(text))


# ---------- System prompt (same one the model was fine-tuned with) ----------

LANGUAGE_INSTRUCTIONS = {
    "english": "Respond in clear, simple English.",
    "hindi": "Respond in Hindi, written in Devanagari script.",
    "hinglish": ("Respond in Hinglish - natural Hindi-English mix in Roman script. "
                 "Keep technical terms (IS numbers, scheme names, product names) in English."),
}

SYSTEM_PROMPT = """You are the BIS AI Assistant, an official-style helper for the \
Bureau of Indian Standards (BIS). You help industries and consumers understand \
applicable Indian Standards, certification schemes, and licensing steps.

STRICT RULES:
1. If RETRIEVED DATA is provided, answer ONLY using that context. Never use \
outside knowledge about specific BIS standards, even if you think you know more.
2. If RETRIEVED DATA says "NONE", the query didn't match a specific product. \
If it's a greeting/small talk, respond warmly and explain what you can help \
with. If it looks like a real product question with no match, say so honestly.
3. If RETRIEVED DATA says "AMBIGUOUS", multiple products matched a broad \
category term - list the option names briefly and ask the user which one \
they mean. Do not pick one for them.
4. Always mention the exact IS standard number and scheme type when available.
5. Keep answers concise and practical for a small business owner or consumer.
6. Present certification steps in order when explaining them.
7. FORMATTING - use these when they genuinely help, not for every reply:
   - Use a markdown table (| Column | Column |) when comparing 2+ products, \
schemes, or standards side by side.
   - Use a chart when showing a genuine numeric comparison (timelines, fees, \
counts). Output a fenced code block labeled "chart" containing ONLY valid \
JSON in this exact shape:
     ```chart
     {"type": "bar", "title": "Short Title", "labels": ["A", "B"], "datasets": [{"label": "Series", "data": [10, 20]}]}
     ```
     Valid "type" values: "bar", "line", "pie". Never invent numbers to fill a chart.
   - For plain single-product lookups, just write normal prose.
"""


def build_language_directive(query):
    if LANGUAGE_MODE != "auto":
        return f"LANGUAGE FOR THIS REPLY: {LANGUAGE_INSTRUCTIONS[LANGUAGE_MODE]}"
    if contains_devanagari(query):
        return "LANGUAGE FOR THIS REPLY: The user wrote in Hindi (Devanagari script). Respond in Hindi, Devanagari script."
    return ("LANGUAGE FOR THIS REPLY: The user's message uses Roman/Latin letters "
            "(English or Hinglish), NOT Devanagari script. Your ENTIRE reply must "
            "also be written in Roman/Latin letters. Do NOT use Devanagari script "
            "anywhere in this reply, even if the topic feels like it calls for "
            "Hindi - stay in Roman letters (English or Hinglish, matching the "
            "user's tone).")


def ask_llm(history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    response = ollama.chat(model=MODEL_NAME, messages=messages, options={"temperature": 0.2})
    return response["message"]["content"]


def offline_answer(query, candidates, options, context_block=None):
    """used only when Ollama isn't reachable - plain but correct"""
    hinglish = bool(_re.search(r"\b(ka|ke|ki|kya|hai|batao|liye|kaise)\b", query.lower()))
    if options:
        names = ", ".join(f"{o['product_name']} ({o['is_standard']})" for o in options[:5])
        return f"Several products match. Which one do you mean: {names}?"
    if (not candidates or candidates[0][1] < CONFIDENCE_THRESHOLD) and context_block and context_block.startswith("Certification Scheme"):
        return context_block.replace("\n", "  \n")
    if not candidates or candidates[0][1] < CONFIDENCE_THRESHOLD:
        return ("Hello! Tell me a product name (like 'pressure cooker' or 'LED bulb') and I'll find its "
                "IS standard, scheme and certification steps.")
    p = candidates[0][0]
    if is_follow_up(query):
        steps = _workflow_steps.get(p["scheme"], [])
        return (f"For **{p['product_name']}** ({p['is_standard']}, {p['scheme']}):  \n" +
                "  \n".join(f"{i+1}. {x}" for i, x in enumerate(steps)) if steps else
                f"**{p['product_name']}** is voluntary - no mandatory steps.")
    if contains_devanagari(query):
        need = "अनिवार्य" if p["mandatory"] else "स्वैच्छिक"
        return f"**{p['product_name']}** के लिए **{p['is_standard']}** मानक लागू है, {p['scheme']} योजना ({need})।"
    if hinglish:
        need = "mandatory" if p["mandatory"] else "voluntary"
        return f"Aapke **{p['product_name']}** ke liye **{p['is_standard']}** lagu hota hai, {p['scheme']} scheme ({need}). {p['scope_summary']}"
    need = "mandatory" if p["mandatory"] else "voluntary"
    return (f"**{p['product_name']}** falls under **{p['is_standard']}** - {p['scheme']} scheme ({need}). "
            f"{p['scope_summary']} Legal basis: {p['legal_basis']}.")


# ---------- threshold calibration (embeddings only) ----------
AMBIG_GAP = 0.04
if _emb_model is not None:
    _in = ["pressure cooker", "steel bars for construction", "LED lamp", "toothpaste", "helmet for bike",
           "packaged drinking water", "electric iron", "cement for house", "toys", "PVC cable"]
    _off = ["how to cook biryani", "cricket score", "write a poem", "bitcoin price", "movie tickets",
            "weather today", "tell me a joke", "song lyrics", "train timings", "pizza near me"]
    _top = lambda q: semantic_candidates(retrieval_query(q), 1)[0][1]
    _in_s, _off_s = sorted(_top(q) for q in _in), sorted(_top(q) for q in _off)
    CONFIDENCE_THRESHOLD = round((_in_s[2] + _off_s[-2]) / 2, 3)    # between weak real queries and strong junk
    print(f"Calibrated confidence threshold: {CONFIDENCE_THRESHOLD} (real >= {_in_s[2]:.2f}, junk <= {_off_s[-2]:.2f})")


# ---------- Extra features: checklist, labs, licence check, related ----------

CHECKLIST = {
    "ISI": ["Factory registration / manufacturing licence", "Factory layout & process flow chart",
            "List of test equipment / in-house lab details", "Test report from a BIS-recognised lab",
            "Trade mark registration (if any)", "Licence fee payment"],
    "CRS": ["Test report from a BIS-recognised lab", "Brand / trade mark details", "Model list & factory address",
            "Authorised Indian Representative (foreign makers)", "Registration fee payment"],
    "Hallmarking": ["GST registration", "Jeweller registration on BIS portal", "Item details for assaying"],
    "Scheme-X": ["Test report as per the IS", "Factory & machinery details", "Quality control plan", "Fee payment"],
    "Scheme-IV": ["Self-declaration of conformity", "Test report from recognised lab", "Brand details"],
    "Voluntary-ISI": ["Same as ISI scheme, only if you choose to certify (not legally required)"],
}

# PLACEHOLDER lab list - BIS recognised lab register isn't in our dataset yet
LABS = [("PLACEHOLDER Lab A", "Jaipur", 26.85, 75.80), ("PLACEHOLDER Lab B", "Jaipur", 26.78, 75.82),
        ("PLACEHOLDER Lab C", "Delhi", 28.61, 77.21), ("PLACEHOLDER Lab D", "Mumbai", 19.08, 72.88),
        ("PLACEHOLDER Lab E", "Chennai", 13.08, 80.27), ("PLACEHOLDER Lab F", "Kolkata", 22.57, 88.36)]
CITY = {"jaipur": (26.91, 75.79), "delhi": (28.61, 77.21), "mumbai": (19.08, 72.88), "chennai": (13.08, 80.27),
        "kolkata": (22.57, 88.36), "bengaluru": (12.97, 77.59), "pune": (18.52, 73.86), "hyderabad": (17.39, 78.49)}


def km(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def find_product(name):
    for p in _products:
        if p["product_name"].lower() == (name or "").lower():
            return p
    c = get_candidates(name or "")
    return c[0][0] if c else None


def related(p, k=4):
    base = _re.match(r"(IS\s*\d+)", p["is_standard"])
    same = [x for x in _products if x is not p and base and x["is_standard"].startswith(base.group(1))]
    same += [x for x in _products if x is not p and x["category"] == p["category"] and x not in same]
    return [{"product_name": x["product_name"], "is_standard": x["is_standard"]} for x in same[:k]]


def licence_check(text):
    t = text.upper()
    m = _re.search(r"CM\s*/\s*L\s*[-–]?\s*(\d{4,12})", t)
    if m:
        ok = len(m.group(1)) == 10
        return {"kind": "ISI licence", "value": f"CM/L-{m.group(1)}", "ok": ok,
                "note": "Format looks valid." if ok else "ISI licence numbers have 10 digits after CM/L-."}
    m = _re.search(r"\bR\s*[-–]?\s*(\d{8})\b", t)
    if m:
        return {"kind": "CRS registration", "value": f"R-{m.group(1)}", "ok": True, "note": "Format looks valid."}
    m = _re.search(r"HUID\W*([A-Z0-9]{6})\b", t)
    if m:
        return {"kind": "Hallmark HUID", "value": m.group(1), "ok": True, "note": "6-character HUID format looks valid."}
    return {"kind": "unknown", "value": "", "ok": False, "note": "No CM/L licence, R-number or HUID found."}


# ---------- official links shown under answers ----------
LINKS = {
    "manak": {"label": "Apply on Manakonline", "url": "https://www.manakonline.in"},
    "crs": {"label": "Register on BIS CRS portal", "url": "https://www.crsbis.in"},
    "standards": {"label": "Search standards on BIS", "url": "https://standards.bis.gov.in"},
    "bis": {"label": "BIS website", "url": "https://www.bis.gov.in"},
    "care": {"label": "Verify on BIS Care app", "url": "https://play.google.com/store/search?q=BIS%20Care&c=apps"},
}


def links_for(p):
    out = []
    if p.get("source_url"):
        out.append({"label": f"View {p['is_standard']} on BIS", "url": p["source_url"]})
    else:
        out.append(LINKS["standards"])
    scheme = str(p.get("scheme", "")).upper()
    out.append(LINKS["crs"] if "CRS" in scheme else LINKS["manak"])
    return out


def scheme_links(text):
    t = text.lower()
    if "crs" in t or "compulsory registration" in t:
        return [LINKS["crs"], LINKS["bis"]]
    if "hallmark" in t or "huid" in t:
        return [LINKS["manak"], LINKS["care"]]
    return [LINKS["manak"], LINKS["bis"]]


# ---------- Analytics (anonymous, no IP stored) ----------

def db():
    con = sqlite3.connect(DB_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS q(ts TEXT, lang TEXT, query TEXT, matched TEXT, outcome TEXT)")
    return con


def log(query, matched, outcome):
    lang = "hi" if contains_devanagari(query) else "en/hinglish"
    with db() as con:
        con.execute("INSERT INTO q VALUES (?,?,?,?,?)",
                    (datetime.datetime.now().isoformat(timespec="seconds"), lang, query[:200], matched, outcome))


# ---------- Follow-ups & general scheme questions ----------

FOLLOW_UP_WORDS = ("step", "process", "procedure", "document", "paper", "kagaz", "fee", "cost", "price", "time",
                   "kitna", "kaise", "how to apply", "apply", "lab", "test", "licence", "license", "renew",
                   "iske", "iska", "uska", "uske", "this", "that", "it ", "same", "aur", "more", "detail",
                   "प्रक्रिया", "दस्तावेज", "इसके", "कैसे")
SCHEME_WORDS = {"hallmark": "Hallmarking", "huid": "Hallmarking", "crs": "CRS", "compulsory registration": "CRS",
                "scheme x": "Scheme-X", "scheme-x": "Scheme-X", "scheme iv": "Scheme-IV", "scheme-iv": "Scheme-IV",
                "isi mark": "ISI", "isi": "ISI", "voluntary": "Voluntary-ISI"}


def is_follow_up(q):
    ql = q.lower() + " "
    return len(ql.split()) <= 9 and any(w in ql for w in FOLLOW_UP_WORDS)


def scheme_context(q):
    ql = q.lower()
    for word, scheme in SCHEME_WORDS.items():
        if word in ql:
            steps = _workflow_steps.get(scheme, [])
            n = sum(1 for p in _products if p["scheme"] == scheme)
            examples = ", ".join([p["product_name"] for p in _products if p["scheme"] == scheme][:4])
            return (f"Certification Scheme: {scheme}\nProducts in database under this scheme: {n}\n"
                    f"Example products: {examples}\nCertification Steps:\n" +
                    "\n".join(f"  {i+1}. {x}" for i, x in enumerate(steps)))
    return None


# ---------- Conversation state (single-user local demo) ----------

_conversation_history = []
_last_matched_candidates = []
MAX_HISTORY_MESSAGES = 20


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/dashboard")
def dashboard():
    return send_from_directory("static", "dashboard.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    global _last_matched_candidates

    data = request.get_json(force=True)
    query = (data.get("message") or "").strip()
    if not query:
        return jsonify({"error": "Empty message."}), 400

    # licence / HUID typed straight into chat
    if _re.search(r"cm\s*/\s*l|huid|\br-\d{8}", query.lower()):
        res = licence_check(query)
        log(query, "", "licence_check")
        return jsonify({"answer": f"**{res['kind']}** {res['value']} — {res['note']} "
                                  "(Format check only. Confirm on the BIS CARE app.)", "licence": res,
                        "links": [LINKS["care"]]})

    candidates = get_candidates(query)
    options = get_disambiguation_options(query, candidates)
    has_confident_match = bool(candidates) and candidates[0][1] >= CONFIDENCE_THRESHOLD

    follow = is_follow_up(query) and _last_matched_candidates and not options
    sc = scheme_context(query)
    if sc and not options and (not has_confident_match or candidates[0][1] < 0.35):
        # "how does CRS work" is about the scheme, not about "Work Chairs"
        follow, has_confident_match, candidates = False, False, []
    if options:
        context_block = "AMBIGUOUS"
    elif has_confident_match and not (follow and candidates[0][1] < 0.3):
        context_block = build_context_block(candidates, _workflow_steps)
        _last_matched_candidates = candidates
    elif sc and not has_confident_match:
        context_block = sc
    elif follow:
        # "steps batao", "documents?" -> about the product we were just discussing
        context_block = build_context_block(_last_matched_candidates, _workflow_steps, limit=1)
        candidates, has_confident_match = _last_matched_candidates, True
    else:
        context_block = scheme_context(query)      # "what is hallmarking?" etc. (None if not a scheme question)

    context_text = context_block if context_block else "NONE"
    user_message_content = f"""RETRIEVED DATA:
{context_text}

{build_language_directive(query)}

USER MESSAGE:
{query}

Respond following the system rules and the LANGUAGE FOR THIS REPLY instruction above."""

    engine = MODEL_NAME
    try:
        if ollama is None:
            raise ConnectionError("ollama not installed")
        answer = ask_llm([{"role": "user", "content": user_message_content}])
    except Exception as e:
        print("[llm] using offline answer:", e)
        answer = offline_answer(query, candidates if has_confident_match else [], options, context_block)
        engine = "offline-template"

    _conversation_history.append({"role": "user", "content": query})
    _conversation_history.append({"role": "assistant", "content": answer})
    del _conversation_history[:-MAX_HISTORY_MESSAGES]

    top = candidates[0] if has_confident_match and not options else None
    product = top[0] if top else None
    log(query, product["is_standard"] if product else "",
        "ambiguous" if options else ("answered" if product else "no_match"))

    result = {"answer": answer, "engine": engine,
              "matched_products": [p["product_name"] for p, s in candidates[:3]] if has_confident_match and not options else []}
    if product and not options:
        result["product"] = product
        result["related"] = related(product)
        result["links"] = links_for(product)
    elif context_block and context_block.startswith("Certification Scheme"):
        result["links"] = scheme_links(query)
    if options:
        result["options"] = options
    return jsonify(result)


@app.route("/api/action", methods=["POST"])
def action():
    """buttons under an answer: steps / checklist / labs / form / related"""
    b = request.get_json(force=True)
    p = find_product(b.get("product"))
    if not p:
        return jsonify({"error": "product not found"}), 404
    kind = b.get("action")
    log(kind, p["is_standard"], f"action_{kind}")
    portal = [LINKS["crs"] if "CRS" in str(p["scheme"]).upper() else LINKS["manak"]]
    if kind == "steps":
        return jsonify({"links": portal, "title": f"{p['scheme']} steps", "items": _workflow_steps.get(p["scheme"]) or
                        ["Voluntary - no mandatory process. Apply on Manakonline if you want the ISI Mark."]})
    if kind == "checklist":
        return jsonify({"links": portal, "title": f"Documents for {p['product_name']}", "items": CHECKLIST.get(p["scheme"], CHECKLIST["ISI"])})
    if kind == "labs":
        lat, lon = CITY.get((b.get("city") or "jaipur").lower(), CITY["jaipur"])
        near = sorted(LABS, key=lambda l: km(lat, lon, l[2], l[3]))[:3]
        return jsonify({"title": "Nearest labs (demo list)", "items": [f"{n}, {c} — {km(lat, lon, a, o):.0f} km" for n, c, a, o in near]})
    if kind == "form":
        return jsonify({"links": portal, "title": "Application draft", "fields": [
            ["Product", p["product_name"], True], ["Indian Standard", p["is_standard"], True],
            ["Scheme", p["scheme"], True], ["Mandatory", "Yes" if p["mandatory"] else "No", True],
            ["Legal basis", p["legal_basis"], True], ["Company name", "", False], ["GSTIN", "", False],
            ["Factory address", "", False]]})
    if kind == "related":
        return jsonify({"links": [LINKS["standards"]], "title": "Related standards", "items": [f"{r['product_name']} — {r['is_standard']}" for r in related(p, 6)]})
    return jsonify({"error": "unknown action"}), 400


@app.route("/api/form.pdf")
def form_pdf():
    p = find_product(request.args.get("product"))
    if not p:
        return "not found", 404
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        return "pip install reportlab", 500
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); y = 790
    c.setFont("Helvetica-Bold", 14); c.drawString(50, y, "BIS certification application - pre-filled draft"); y -= 18
    c.setFont("Helvetica", 9); c.drawString(50, y, f"BIS Assistant, {datetime.date.today():%d %b %Y}. Final submission on Manakonline."); y -= 30
    rows = [("Product", p["product_name"]), ("Indian Standard", p["is_standard"]), ("Scheme", p["scheme"]),
            ("Mandatory", "Yes" if p["mandatory"] else "No"), ("Legal basis", p["legal_basis"]),
            ("Company name", request.args.get("company", "")), ("GSTIN", ""), ("Factory address", "")]
    for k, v in rows:
        c.setFont("Helvetica-Bold", 10); c.drawString(50, y, k); c.setFont("Helvetica", 10)
        c.drawString(190, y, (v or "_" * 40)[:75]); y -= 20
    y -= 10; c.setFont("Helvetica-Bold", 11); c.drawString(50, y, "Documents to keep ready"); y -= 18
    c.setFont("Helvetica", 10)
    for d in CHECKLIST.get(p["scheme"], CHECKLIST["ISI"]):
        c.rect(50, y - 2, 8, 8); c.drawString(65, y, d); y -= 16
    c.save()
    return send_file(io.BytesIO(buf.getvalue()), mimetype="application/pdf", as_attachment=True,
                     download_name=f"BIS_draft_{p['is_standard'].replace(' ', '_')}.pdf")


@app.route("/api/scan", methods=["POST"])
def scan():
    f = request.files.get("image")
    text = ""
    if f:
        try:
            import cv2, numpy as np
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        except Exception as e:
            print("[scan]", e)
    res = licence_check(text) if text else {"kind": "unknown", "value": "", "ok": False,
                                            "note": "Couldn't read a QR code - try a clearer photo or type the number."}
    log("qr scan", "", "scan")
    return jsonify({"answer": f"**{res['kind']}** {res['value']} — {res['note']} (Confirm on the BIS CARE app.)",
                    "links": [LINKS["care"]]})


@app.route("/api/audit", methods=["POST"])
def audit():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "upload a csv"}), 400
    rows = list(csv.reader(io.StringIO(f.read().decode("utf-8-sig", errors="ignore"))))
    head = [h.strip().lower() for h in rows[0]] if rows else []
    col = head.index("product") if "product" in head else 0
    out = []
    for r in rows[1:] if "product" in head else rows:
        if len(r) <= col or not r[col].strip():
            continue
        c = get_candidates(r[col])
        if not c or c[0][1] < CONFIDENCE_THRESHOLD:
            out.append({"product": r[col], "is": "", "scheme": "", "status": "Not found - check manually"})
        else:
            p = c[0][0]
            out.append({"product": r[col], "is": p["is_standard"], "scheme": p["scheme"],
                        "status": "Mandatory" if p["mandatory"] else "Voluntary", "matched": p["product_name"]})
    log(f"bulk audit ({len(out)})", "", "bulk_audit")
    return jsonify({"rows": out[:500]})


@app.route("/api/stats")
def stats():
    with db() as con:
        q = lambda s: con.execute(s).fetchall()
        return jsonify({
            "total": q("SELECT COUNT(*) FROM q")[0][0],
            "outcomes": q("SELECT outcome, COUNT(*) FROM q GROUP BY outcome ORDER BY 2 DESC"),
            "langs": q("SELECT lang, COUNT(*) FROM q GROUP BY lang"),
            "top": q("SELECT matched, COUNT(*) FROM q WHERE matched!='' GROUP BY matched ORDER BY 2 DESC LIMIT 8"),
            "gaps": q("SELECT query, COUNT(*) FROM q WHERE outcome='no_match' GROUP BY query ORDER BY 2 DESC LIMIT 8"),
            "recent": q("SELECT ts, query, matched, outcome FROM q ORDER BY ts DESC LIMIT 12")})


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    global _last_matched_candidates
    _conversation_history.clear()
    _last_matched_candidates = []
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    ok = False
    if ollama:
        try:
            names = [m.get("model", m.get("name", "")) for m in ollama.list().get("models", [])]
            ok = any(n.startswith(MODEL_NAME) for n in names)
            print(f"Ollama models: {names}")
        except Exception as e:
            print("Ollama not reachable:", e)
    print(f"Model '{MODEL_NAME}': {'READY' if ok else 'NOT FOUND -> run: ollama create bis-assistant -f Modelfile (offline template mode until then)'}")
    print("Open http://127.0.0.1:5000   |   analytics: http://127.0.0.1:5000/dashboard")
    app.run(debug=False, port=5000)
