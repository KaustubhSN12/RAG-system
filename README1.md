<<<<<<< HEAD
# Beginner RAG Pipeline — Solar System Q&A

A minimal, fully working Retrieval-Augmented Generation (RAG) system you can
run locally to learn how RAG actually works under the hood. Topic: basic
solar system facts, so you can focus on the *mechanics* instead of the data.
=======
# RAG-system

Topic: basic
solar system facts

>>>>>>> 79d0460057e6633eb2db04fb8b05ff7c0dc84269

## What is RAG, in one paragraph

A plain LLM only knows what it learned during training and can hallucinate
facts. RAG fixes this by first **retrieving** relevant text from your own
documents (via similarity search over embeddings), then **augmenting** the
LLM's prompt with that retrieved text, so the model **generates** an answer
grounded in real source material instead of guessing.

<<<<<<< HEAD
## Project structure

```
rag_learning_project/
=======

## Project structure

```
RAG/
>>>>>>> 79d0460057e6633eb2db04fb8b05ff7c0dc84269
├── data/
│   └── solar_system_facts.txt   # knowledge base (10 short facts)
├── 1_ingest.py                  # builds the vector index (run once)
├── 2_query.py                   # ask questions against the index
├── requirements.txt
├── .env.example                 # copy to .env and add your API key (optional)
└── index/                       # created automatically by 1_ingest.py
    ├── faiss_index.bin
    └── chunks.json
```
<<<<<<< HEAD

## How to run it in VS Code

### 1. Open the folder
Open `rag_learning_project` as a folder in VS Code, then open a terminal
=======
### 1. Open the folder
Open `RAG` as a folder in VS Code, then open a terminal
>>>>>>> 79d0460057e6633eb2db04fb8b05ff7c0dc84269
(`` Ctrl+` ``).

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate  # venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Build the vector index
```bash
python 1_ingest.py
```
This reads `data/solar_system_facts.txt`, splits it into chunks, embeds each
chunk with a free local model (`all-MiniLM-L6-v2`), and saves a FAISS index
plus the chunk text into the `index/` folder. The model (~90MB) downloads
automatically the first time — needs internet access once.

### 5. Ask questions
```bash
python 2_query.py
```
Try questions like:
- "Which planet is closest to the Sun?"
- "Why does Venus have a runaway greenhouse effect?"
- "What is the Great Red Spot?"

For each question you'll see the retrieved chunks (with similarity scores)
printed first — that's retrieval working — followed by a generated answer if
you've set up an LLM key.

<<<<<<< HEAD
=======

>>>>>>> 79d0460057e6633eb2db04fb8b05ff7c0dc84269
## Enabling the "generation" step (optional but recommended)

Retrieval works with zero API keys. To also generate a natural-language
answer, add an OpenAI key:

1. Copy `.env.example` to `.env`
2. Add your key: `OPENAI_API_KEY=sk-...`

<<<<<<< HEAD
or pip install google-genai

=======
>>>>>>> 79d0460057e6633eb2db04fb8b05ff7c0dc84269
Without a key, `2_query.py` still runs and shows you exactly which chunks
were retrieved — useful for understanding retrieval in isolation before
paying for generation.

<<<<<<< HEAD
=======

>>>>>>> 79d0460057e6633eb2db04fb8b05ff7c0dc84269
## What to tweak once it's running (learning exercises)

- **Change `TOP_K`** in `2_query.py` (try 1 vs 5) and see how the answer
  quality changes.
- **Add more facts** to `data/solar_system_facts.txt` (any blank-line
  separated paragraph becomes a new chunk) and re-run `1_ingest.py`.
- **Swap the embedding model** — try `all-mpnet-base-v2` (slower, more
  accurate) instead of `all-MiniLM-L6-v2`.
- **Break it on purpose**: ask a question with no answer in the data (e.g.
  "What is Pluto's atmosphere made of?") and watch the model say it doesn't
  know instead of hallucinating — that's the grounding effect of RAG.
- **Print raw similarity scores** for a bad match vs a good match to build
  intuition for what "similarity" means numerically.

<<<<<<< HEAD
=======

>>>>>>> 79d0460057e6633eb2db04fb8b05ff7c0dc84269
## Why each library is used

| Library | Role |
|---|---|
| `sentence-transformers` | Turns text into vectors (embeddings) locally, free |
| `faiss` | Fast similarity search over those vectors |
| `numpy` | Array handling glue between the two |
| `openai` | The "G" in RAG — generates the final answer |
| `python-dotenv` | Loads your API key from `.env` instead of hardcoding it |
