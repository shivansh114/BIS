# BIS Assistant

**An AI assistant for Indian Standards and BIS services** — describe your product in your own words, and it finds the right Indian Standard, tells you whether certification is mandatory, and links you to the official next step.

Built by **Team Astra** for **Smart India Hackathon 2026 — Problem Statement 26107**
*(AI-powered Intelligent Assistant for Indian Standards and BIS Services for Industries and Consumers)*

▶ **Demo video:** https://youtu.be/3YmM2G0tWlw

---

## The problem

BIS has published **23,813 Indian Standards**. To find which one applies to a product — and whether certification is mandatory, which scheme applies and what steps to follow — people have to search through many documents, portals and PDFs. This is slow and confusing, especially for MSMEs, startups and consumers, and many end up paying consultants for a single answer.

## What it does

- **Ask in your own words** — English, Hindi (हिन्दी) or Hinglish, by typing or speaking
- **Asks a clarifying question** when a description fits several standards (e.g. *"I make cables"* → welding / vehicle / PVC-insulated / elastomer cables)
- **Gives the exact standard** (IS number), the certification scheme (ISI, CRS, Hallmarking, …) and whether it is **mandatory**, with the legal basis
- **Shows the next official step** — certification steps, document checklist, and direct links to **Manakonline**, the **CRS portal** or **BIS Care**
- **Pre-fills an application draft** (PDF) with everything it already knows
- **Licence / HUID format check** (typed or scanned from a QR code) with hand-off to the official **BIS Care** app
- **Bulk audit** — upload a CSV of products and get the standard and status for each
- **Declines off-topic questions** instead of making things up
- **Runs fully offline** — our fine-tuned model runs on a local machine; no cloud AI API is used
- **Embeddable** — designed to sit on the BIS website as a chat widget; no app, no login

## How it works

```
User question (EN / HI / Hinglish)
   │
   ├─ 1. Understand   language detection · Hindi → English terms · everyday names → BIS terms · spelling correction
   ├─ 2. Search       typo-tolerant keyword search (TF-IDF, word + character) over all 23,813 standards
   │                  optional: hybrid semantic search (multilingual embeddings + keywords)
   ├─ 3. Decide       clear match → answer · several close matches → ask · weak / off-topic → decline
   ├─ 4. Rules        scheme, mandatory status, steps and documents come from BIS records — not from the AI
   └─ 5. Word & link  fine-tuned Llama 3.1 8B writes the reply in the user's language; official links are attached
```

**Key design choice:** the standard and the legal facts are decided by search + rules over BIS data. The language model only explains them — so every fact in an answer can be traced to a record.

## Results (measured on our laptop, 4 Oct 2026)

| Test | Result |
|---|---|
| 50 hand-written product questions — right standard (keyword search, full catalogue) | **38 / 50 (76%)** |
| Broad words that should trigger a clarifying question | **14 / 15** |
| Fine-tuned model vs base Llama 3.1 8B — right IS number in the reply (27 questions) | **88% vs 66%** |
| Made-up IS numbers in replies | **0%** (both models) |

Reproduce with `eval_assistant.py` and `compare_models.py` (see below). The answer key is in `eval_questions.csv`; a second set of 40 unseen questions is in `eval_heldout.csv`.

**Known limitations:** off-topic filtering on the full catalogue still lets some unrelated questions through (14/20 declined); short or very vague product names are the weakest case; the lab finder uses placeholder data (the BIS recognised-lab register is not in our dataset); licence checks verify the *format* only — official verification is done in BIS Care.

## Tech stack

Python · Flask · scikit-learn (TF-IDF) · sentence-transformers (optional) · Ollama · Llama 3.1 8B + LoRA (Unsloth, 4-bit) · ReportLab (PDF) · OpenCV (QR decode) · HTML / CSS / JavaScript · Web Speech API

## Project structure

```
app_local.py              Flask app: search, rules, chat API, actions, PDF form, scan, bulk audit, analytics
static/index.html         demo BIS-style website with the embedded chat assistant
static/dashboard.html     anonymised usage dashboard for BIS
bis_full_dataset.json     curated product → standard / scheme mapping
eval_assistant.py         search evaluation (hand-written questions)
eval_questions.csv        50-question answer key
eval_heldout.csv          40 unseen questions
compare_models.py         fine-tuned vs base model comparison
Modelfile                 imports the fine-tuned GGUF into Ollama as "bis-assistant"
finetune_lora.py          LoRA fine-tuning of Llama 3.1 8B (Unsloth)
export_gguf.py            export the adapter to GGUF for Ollama
generate_training_data.py builds the fine-tuning examples
START_DEMO.bat            one-click start on Windows
DEMO_SCRIPT.md            2-minute live demo walkthrough
```

## Running it (Windows)

**Requirements:** Python 3.10+, [Ollama](https://ollama.com), and the BIS catalogue data folder (`standards_enriched.csv`, collected from the published BIS standards list — not included in this repo because of size).

1. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Load the fine-tuned model into Ollama** (once) — place the GGUF in `bis-assistant-lora/`, then:
   ```powershell
   ollama create bis-assistant -f Modelfile
   ```
3. **Point to the catalogue** — edit `START_DEMO.bat` so `BIS_DATA_DIR` is the folder containing `standards_enriched.csv`.
4. **Start** — double-click `START_DEMO.bat`, then open **http://127.0.0.1:5000** and click **Ask BIS Assistant**.
   Analytics dashboard: http://127.0.0.1:5000/dashboard

If Ollama is not running, the app still answers with simple template replies, so it never breaks.

**Search mode:** `START_DEMO.bat` sets `BIS_EMBED_MODEL=none` (keyword search — our best-scoring mode). Remove that line to try hybrid semantic search; the embedding model downloads once and the catalogue is embedded and cached on first start.

## Evaluation

```powershell
$env:BIS_DATA_DIR = "D:\path\to\catalogue"
python eval_assistant.py eval_questions.csv     # 50 hand-written questions
python eval_assistant.py eval_heldout.csv       # 40 unseen questions
python compare_models.py                        # fine-tuned vs base Llama (needs both models in Ollama)
```

## Data sources

- BIS published standards list and standard detail pages (23,813 standards)
- BIS certification scheme lists (Scheme I / II / IV / X, Hallmarking)
- Quality Control Order notifications (Gazette of India)
- BIS licensing guidelines

Classification is copied from BIS records and may lag the latest QCO — always confirm on the official BIS portals.

## Team

**Team Astra** · Poornima University · Smart India Hackathon 2026

---

*This is a hackathon prototype. It is not an official BIS service; for official information, licences and verification, use [bis.gov.in](https://www.bis.gov.in), [Manakonline](https://www.manakonline.in) and the BIS Care app.*
