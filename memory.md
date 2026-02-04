# AI Data Expert V1

## Project Info
- **Author:** Munir Rodriguez
- **Created:** 01/29/2026
- **Repository:** https://github.com/munirche/AI-Data-Expert-V1-repo.git

## Description
A human-in-the-loop AI system that learns from expert annotations to eventually automate data analysis.

### Objective
1. Receives as input a large, complex data set
2. A specialist detects known patterns, tags them in the data, documents findings in natural language and with technical comments
3. The process is repeated over and over, with AI learning what the analyst detects and documents
4. Finally, when trained, the model will be able to produce a suggested analysis based on what it has learned

## Software Prerequisites

| Package | Install Command | Purpose |
|---------|-----------------|---------|
| google-genai | `pip install google-genai` | LLM API calls (Gemini) |
| lancedb | `pip install lancedb` | Vector database for RAG |
| Docker Desktop | [docker.com](https://www.docker.com/products/docker-desktop/) | Container runtime for tools |

### Lance Data Viewer

A Docker-based tool for inspecting LanceDB tables visually in the browser.

**To run:**
```powershell
docker run --rm -p 8080:8080 -v "C:/Users/munir/Projects/AI Data Expert V1/expert_learning_system_v1_db:/data:ro" ghcr.io/gordonmurray/lance-data-viewer:lancedb-0.24.3
```

**Then open:** http://localhost:8080

**To stop:** Press `Ctrl+C` in PowerShell

## Environment Variables

| Variable | Purpose | How to Set (Windows) |
|----------|---------|---------------------|
| GEMINI_API_KEY | Google Gemini API authentication | `[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your-key', 'User')` |

## Preferences

| Setting | Value | Notes |
|---------|-------|-------|
| LLM for trials | Google Gemini API (free tier) | Using `gemini-2.0-flash` model |
| Vector DB | LanceDB | Compatible with Python 3.14 |
| Embeddings | Gemini text-embedding-004 | Via LanceDB integration |

## Research

### System Type
A **Human-in-the-Loop Active Learning System** where:
- Specialists annotate data and document patterns in natural language
- System learns progressively from annotations
- AI eventually automates and produces analyses on its own

### Three Approaches (Start to Advanced)

#### 1. RAG-Based (Recommended Start)
- Experts annotate 50-100 examples
- Store in vector database with natural language documentation
- AI retrieves similar past analyses for new datasets
- No retraining needed - just update knowledge base
- Achieves 80-90% accuracy with 200-500 examples

#### 2. Active Learning + Fine-Tuning
- AI identifies most uncertain cases for expert review
- Reduces expert workload by 60-80%
- Progressively fine-tune model on annotations
- Best for recurring pattern types

#### 3. Hybrid (Advanced)
- Combines RAG + fine-tuned models + rules
- Best accuracy but more complex

### The Learning Loop
```
Expert annotates → Store in vector DB → New dataset arrives →
Retrieve similar examples → AI generates analysis →
Expert corrects → Store corrections → Repeat (improves over time)
```

### Tools to Consider
| Category | Options | Current Choice |
|----------|---------|----------------|
| Vector DB | ChromaDB, Pinecone, LanceDB | **LanceDB** (Python 3.14 compatible) |
| LLM | GPT-4, Claude, Gemini, Llama-3 | **Gemini 2.0 Flash** (free tier) |
| Annotation | Prodigy ($390), Label Studio (free) | TBD |
| Framework | LangChain, LlamaIndex | Custom (for learning) |

### Estimated Timeline
- **Weeks 1-4:** Expert labels initial examples (50-200)
- **Weeks 5-8:** AI assists, expert corrects
- **Weeks 9-12:** AI handles 70%+ autonomously
- **Week 13+:** Expert reviews only edge cases

### Next Steps
- Start with RAG approach using 50 annotations
- Measure results, then scale what works

## Code Standards

### Commenting
**Level: Moderate**
- Docstrings for all functions (purpose, parameters, return values)
- Inline comments for complex or non-obvious sections
- No comments for self-explanatory code

## Project Structure
```
AI Data Expert V1/
├── Docs/
│   ├── expert_learning_system_guide.md              # Research guide by Claude
│   ├── concepts.md                                  # Key concepts reference
│   ├── how_it_works_expert_learning_system_v1.md    # V1 explanation + what code does
│   ├── how_it_works_export_db_v1.md                 # Export utility documentation
│   ├── how_it_works_launch_lance_viewer.md          # Lance viewer documentation
│   ├── how_it_works_cli_v2.md                       # V2 CLI documentation
│   ├── Discussion_expert_learning_system_v1.md      # Design notes and improvements
│   └── Plan_for_expert_learning_system_v2.md        # V2 implementation plan
├── Code/
│   ├── expert_learning_system_v1.py                 # V1 RAG demo (reference)
│   ├── expert_learning_system_v2.py                 # V2 generic RAG engine
│   ├── cli_v2.py                                    # V2 CLI interface
│   ├── export_db_v1.py                              # Database export utility (CSV/JSON)
│   └── launch_lance_viewer.py                       # Opens Lance Data Viewer in browser
├── corpus/
│   └── corpus.csv                                   # Ground truth test cases (15 cases)
├── config.json                                      # Use case configuration
├── expert_learning_system_v1_db/                    # V1 LanceDB database (not in git)
├── expert_learning_system_v2_db/                    # V2 LanceDB database (not in git)
├── memory.md
├── requirements.txt                                 # Python dependencies
└── .gitignore
```

### File Naming Convention
Scripts, databases, and documentation are versioned together:
- Script: `<name>_v1.py`
- Database: `<name>_v1_db/`
- Documentation: `how_it_works_<name>_v1.md`

This allows multiple versions to coexist during development.

## Time Log

| Date       | Hours | Description                              |
|------------|-------|------------------------------------------|
| 01/30/2026 | 2     | Project setup, research, RAG concepts    |
| 01/31/2026 | 2     | First functional script (expert_learning_system.py) |
| 02/01/2026 | 2.5   | File versioning, export utility, API fixes, custom embeddings |
| 02/01/2026 | 1     | Discussion on possible improvements |
| 02/02/2026 | 2     | Analyzing and planning V2, Lance Data Viewer setup |
| 02/02/2026 | 1.5   | V2 plan document, CLI commands, use case design |
| 02/02/2026 | 0.2   | Phase 1 implementation: engine, CLI, corpus |
| 02/02/2026 | 0.8   | Phase 2 testing, analyze command, documentation |

**Total: 12 hours**

## Notes
- **01/29/2026:** Project initialized. Code folder contains RAG implementation from Claude research session - needs review, understanding, and testing before use.
- **02/01/2026:**
  - Added V1 versioning to scripts, databases, and documentation
  - Created `export_db_v1.py` utility for CSV/JSON exports
  - Fixed API key compatibility (GEMINI_API_KEY → GOOGLE_API_KEY)
  - Added `dataset_details` field to store raw data with annotations
  - Replaced deprecated `google-generativeai` with custom embedding using `google-genai`
- **02/02/2026:**
  - Installed Docker and Lance Data Viewer for database inspection
  - Planned and designed V2: separation of engine from use case, CLI-first approach
  - Implemented V2 Phase 1: generic engine, CLI with load/list/stats/reset/search/add/analyze
  - Created config.json for patient bloodwork use case
  - Created corpus with 15 test cases
  - Enhanced lance viewer to support multiple databases
  - Phase 2 mostly complete: analyze command with compare and batch modes working
  - Remaining: expand corpus to 50 cases, measure baseline accuracy
  - Added `requirements.txt` for dependency management

## Possible Improvements

1. **Embedding structure:** Currently only `summary + analysis` is embedded. Evaluate and test different structures:
   - With record data fields included (e.g., "age: 55, glucose: 94, ...")
   - Without record data fields (current)
   - Measure retrieval quality differences between approaches

2. **Similar examples selection:** Currently fixed at 3 examples. Consider:
   - Similarity threshold (only include cases above e.g., 60% similarity)
   - Dynamic count based on database size
   - Configurable via CLI flag or config.json
   - Combined approach: "Top N, but only if similarity > X%"

3. **Local embeddings:** Research and possibly adopt local embeddings (e.g., `sentence-transformers`, Ollama) so the vector store stops depending on Gemini's embedding API. Evaluate trade-offs: latency, accuracy, cost.

4. **Multi-use-case configuration:** Extract settings to config files per use case, allowing multiple domains to coexist:
   ```
   projects/
   ├── patient_bloodwork/
   │   ├── config.json
   │   └── database/
   ├── revenue_analysis/
   │   ├── config.json
   │   └── database/
   ```

