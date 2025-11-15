# 📘 RAG Project (LangChain + Chroma + Groq + BGE + OCR)

A simple local RAG pipeline that processes PDF/TXT files (including scanned PDFs with OCR), creates BGE embeddings, stores them in Chroma, and answers questions using Groq LLM.

---

## 📂 Project Structure
```
project_root/
│
├── main.py
├── src/
├── config/settings.yaml
├── data/
│   ├── raw/          ← put your PDFs/TXT files here
│   ├── chunks/       ← auto-generated
│   ├── chroma/       ← auto-generated
│   └── manifests/    ← auto-generated
├── .env
├── .env.template
├── requirements.txt
└── README.md
```

---

## 🚀 1. Setup
### Create a virtual environment
```
python -m venv .venv
```

### Activate it
**Windows:**
```
.venv\Scripts\activate
```

**macOS / Linux:**
```
source .venv/bin/activate
```

### Install dependencies
```
pip install -r requirements.txt
```

---

## 🔑 2. Add API Keys
Create your `.env` file:
```
cp .env.template .env
```

Open `.env` and add:
```
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_TOKEN=your_hf_token_optional
```
Groq key is **required**. HuggingFace token is **optional** but recommended for faster model downloads.

---

## 📥 3. Add Your Documents
Put all PDF/TXT files into:
```
data/raw/
```
### Rules:
- Max **10 files**
- Max **50 MB per file**
- Replacing a file re-indexes it
- Exceeding 10 files deletes the oldest automatically

---

## ▶️ 4. Run the Pipeline
Run:
```
python main.py
```
The system will:
- Sync raw files with the manifest
- Extract text (OCR for scanned pages)
- Chunk documents
- Generate embeddings
- Save vectors in Chroma
- Retrieve top-k chunks
- Ask Groq LLM
- Print final answer + citations + text previews

---

## ❓ 5. Ask a Question
Edit this line in **main.py**:
```
query = "Your question here"
```
Then run again:
```
python main.py
```

---

## 📤 6. Outputs (Auto-generated)
- Chunk JSONL → `data/chunks/`
- Chroma vector DB → `data/chroma/`
- Manifest → `data/manifests/manifest.json`

You don’t need to modify or commit these.

---

## 🧹 7. Reset Everything
If you want to start fresh:
```
rm -rf data/chroma/*
rm -rf data/chunks/*
rm -rf data/manifests/manifest.json
```

**Windows PowerShell:**
```
Remove-Item -Recurse -Force data\chroma\*
Remove-Item -Recurse -Force data\chunks\*
Remove-Item -Force data\manifests\manifest.json
```

---

## ✔ Notes
- **BGE model:** BAAI/bge-base-en-v1.5
- **OCR:** TrOCR (microsoft/trocr-base-printed)
- **LLM:** Groq — llama-3.3-70b-versatile
- **Vector DB:** Chroma (local)
- **Language pipeline:** LangChain

