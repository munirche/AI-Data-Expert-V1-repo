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
│   ├── expert_learning_system_guide.md # Research guide by Claude
│   └── concepts.md                     # Key concepts reference
├── Code/
│   └── expert_learning_system.py       # RAG implementation using Gemini + LanceDB
├── memory.md
└── .gitignore
```

## Time Log

| Date       | Hours | Description                              |
|------------|-------|------------------------------------------|
| 01/30/2026 | 2     | Project setup, research, RAG concepts    |

**Total: 2 hours**

## Notes
- **01/29/2026:** Project initialized. Code folder contains RAG implementation from Claude research session - needs review, understanding, and testing before use.

