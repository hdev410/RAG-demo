# Minimal RAG Demo

See [SKILL.md](SKILL.md) for the architecture, folder guide, and commands.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m rag.ingestion.build_index
python app.py
```
