# Concepts

A reference for key concepts encountered during research and development.

---

## Index

1. [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
   - [Embeddings](#embeddings)
   - [Augment Phase](#augment-phase)
   - [Generate Phase](#generate-phase)
2. [LLM Calls via API: Possible Approaches](#llm-calls-via-api-possible-approaches)
3. [Realistic Limits of Current RAG Implementation](#realistic-limits-of-current-rag-implementation)
4. [External Medical Data Sources](#external-medical-data-sources)
   - [EMR Integration via API](#emr-integration-via-api)
   - [Public Medical Databases](#public-medical-databases)

---

## RAG (Retrieval-Augmented Generation)

Instead of an AI answering from memory alone, it first **looks up relevant information** and then **generates an answer using that information**.

### Analogy
Like an open-book exam:
- **Without RAG**: Answer from memory only (might forget or hallucinate)
- **With RAG**: Flip through notes to find relevant examples, then write answer based on what you found

### How It Works

```
1. RETRIEVE    →    2. AUGMENT    →    3. GENERATE
   Find similar         Add retrieved       AI creates response
   past examples         info to prompt      using the examples
```

1. **Retrieve**: New data comes in → system searches database for similar past cases
2. **Augment**: Retrieved examples get added to the AI's prompt as context
3. **Generate**: AI produces response guided by the real examples

### Why It Works for This Project
- Expert annotates 50-100 examples → stored in database
- New dataset arrives → system finds similar past analyses
- AI says: "Based on how the expert handled similar cases, here's my analysis..."
- Expert corrects if needed → correction gets stored → system improves

### Key Benefit
No retraining needed. Just keep adding expert examples to the database, and the AI automatically uses them.

### RAG Workflow Options

RAG is a tool, not a rigid process. After retrieval, the expert has full control:

**Option A: Expert creates annotation independently**
```
New data → Retrieve similar past cases → Expert VIEWS them as reference
                                              ↓
                                        Expert writes own annotation
                                              ↓
                                        Store new annotation in DB
```
RAG provides **context/inspiration**, but expert does the work.

**Option B: Expert reviews AI suggestion**
```
New data → Retrieve similar past cases → AI generates suggested analysis
                                              ↓
                                        Expert reviews
                                              ↓
                              Accept / Edit / Reject
                                              ↓
                                        Store result in DB
```
RAG provides a **draft**, expert validates.

**Combined Workflow (Recommended)**
```
New dataset arrives
        │
        ▼
Retrieve similar past cases
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  Show AI suggestion              Show retrieved examples
                                     as reference
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
            Expert decides:
            • Accept AI suggestion
            • Edit AI suggestion
            • Write from scratch using examples as guide
            • Write from scratch using own knowledge
                   │
                   ▼
            Store final annotation → DB grows → System improves
```

**How This Evolves Over Time**
- **Early phase**: Expert mostly writes from scratch (builds initial knowledge base)
- **Middle phase**: Expert reviews AI suggestions, corrects often
- **Late phase**: Expert accepts most AI suggestions with minimal edits

---

### Embeddings

Embeddings are **numerical representations** of text (or data) that capture meaning. They allow computers to measure **similarity** between pieces of content.

### Analogy
Imagine converting words into GPS coordinates:
- "Happy" → (10.2, 5.8)
- "Joyful" → (10.1, 5.9) ← very close to "Happy"
- "Angry" → (2.3, 8.1) ← far from "Happy"

Words with similar meanings end up with similar coordinates. The computer can now measure distance between concepts.

### What They Look Like
Text gets converted into a list of numbers (typically 768-1536 numbers):
```
"Revenue declined in Q3" → [0.123, -0.456, 0.789, 0.234, ... ]
                            (hundreds or thousands of values)
```

### Why RAG Needs Embeddings

```
1. STORE: Expert annotation → Convert to embedding → Save in vector database

2. SEARCH: New data arrives → Convert to embedding → Find closest matches

           New data embedding: [0.5, 0.3, 0.8, ...]
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
           Past case A        Past case B      Past case C
           [0.4, 0.3, 0.7]   [0.9, 0.1, 0.2]  [0.5, 0.2, 0.9]
           Distance: 0.14     Distance: 0.85   Distance: 0.11
                                                    ↑
                                              MOST SIMILAR
```

### Key Points
- Embeddings are created by AI models (e.g., OpenAI's text-embedding-3-large)
- Similar content = similar embeddings = small distance between them
- Vector databases are optimized to search millions of embeddings quickly
- You don't create embeddings manually - the embedding model does it for you

### Embedding Model Options

Different providers offer embedding models with varying quality, dimensions, and cost:

| Provider | Model | Dimensions | Notes |
|----------|-------|------------|-------|
| **Google** | text-embedding-004 | 768 | Current choice, free tier |
| **OpenAI** | text-embedding-3-small | 1536 | Good balance |
| **OpenAI** | text-embedding-3-large | 3072 | Highest quality, most expensive |
| **Cohere** | embed-english-v3 | 1024 | Good for search |
| **Voyage AI** | voyage-large-2 | 1536 | High quality, Anthropic recommends |
| **Local** | sentence-transformers | 384-1024 | Free, runs on your machine |
| **Local** | Ollama (nomic-embed) | 768 | Easy local setup |

**Note:** Anthropic/Claude does not offer an embedding API. Claude is only a chat/completion model. If using Claude for generation, pair it with a separate embedding provider (Anthropic recommends Voyage AI).

### Dimensionality Trade-offs

| Dimensions | Pros | Cons |
|------------|------|------|
| **384-512** | Fast, small storage | May miss nuance |
| **768** (current) | Good balance | Standard choice |
| **1536-3072** | Captures more detail | Slower search, more storage, higher cost |

Dimensionality is fixed per model - you choose it by selecting a model. Higher dimensions capture more semantic nuance, but with diminishing returns. 768 is sufficient for most use cases.

### Ways to Improve Embedding Quality

**Better text preparation** - Add context to raw values:
- Weak: "92"
- Better: "glucose: 92 mg/dL (normal range: 70-100)"

**Domain-specific models** - Models trained for specific fields:
- PubMedBERT (medical/clinical)
- FinBERT (financial)
- LegalBERT (legal)

**Hybrid search** - Combine embeddings with keyword matching to catch exact terms that embeddings might miss.

### Cost Considerations

| Model | Cost per 1M tokens |
|-------|-------------------|
| Gemini text-embedding-004 | Free (within limits) |
| OpenAI text-embedding-3-small | ~$0.02 |
| OpenAI text-embedding-3-large | ~$0.13 |
| Local (sentence-transformers) | Free (your compute) |

### Domain-Specific Embeddings: PubMedBERT

For specialized domains, models trained on domain-specific text can outperform general-purpose embeddings.

**PubMedBERT** (Microsoft Research) is trained on 14M+ PubMed abstracts and clinical text. It understands medical terminology natively.

| Aspect | General Models | PubMedBERT |
|--------|----------------|------------|
| Training data | Web, books, Wikipedia | PubMed abstracts, clinical notes |
| Medical vocabulary | Learned incidentally | Built into the model |
| "Creatinine" | Knows it's a word | Knows it's a kidney biomarker |
| "WBC" | May confuse acronyms | Understands it's white blood cell count |

**Available variants:**

| Model | Trained On | Best For |
|-------|------------|----------|
| PubMedBERT-abstract | PubMed abstracts | Research/literature |
| PubMedBERT-fulltext | Full articles | Detailed clinical text |
| ClinicalBERT | MIMIC clinical notes | Real patient records |
| BioBERT | PubMed + PMC | General biomedical |

**How to use:** Available via Hugging Face (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`). Requires `sentence-transformers` library. Runs locally - no API costs, data stays private.

**Trade-offs:**

| Pros | Cons |
|------|------|
| Better medical understanding | Requires local setup |
| Free (no API costs) | Slower than cloud APIs |
| Data stays private | 768 dimensions only |
| Native clinical vocabulary | May underperform on non-medical text |

**When to consider switching:** When scaling beyond initial corpus, seeing retrieval quality issues, working with heavy medical jargon, or needing to keep patient data fully local.

---

### Augment Phase

After retrieval finds similar past cases, the **augment phase** combines everything into a prompt for the AI. This is where context is assembled.

### What Gets Combined

```
┌─────────────────────────────────────────────────────────┐
│                    AUGMENTED PROMPT                     │
├─────────────────────────────────────────────────────────┤
│  1. System instructions                                 │
│     "You are an expert data analyst..."                 │
│                                                         │
│  2. Retrieved examples (from database)                  │
│     Example 1: [Past data summary] → [Expert analysis]  │
│     Example 2: [Past data summary] → [Expert analysis]  │
│     Example 3: [Past data summary] → [Expert analysis]  │
│                                                         │
│  3. New data to analyze                                 │
│     [Current dataset summary or raw data]               │
│                                                         │
│  4. Task instruction                                    │
│     "Analyze this dataset following the same approach"  │
└─────────────────────────────────────────────────────────┘
```

### Concrete Example

```
PROMPT SENT TO AI:
─────────────────────────────────────────────────────────
You are an expert data analyst. Below are examples of how
similar datasets were analyzed in the past.

EXAMPLE 1:
Dataset: Q2 2024 sales data showing 12% decline in Region North
Expert Analysis: "The Q2 decline correlates with supply chain
disruptions reported in April. Recommend investigating vendor
delays. Pattern: seasonal + external factor."

EXAMPLE 2:
Dataset: Q3 2023 revenue showing 18% decline across all regions
Expert Analysis: "Broad Q3 decline reflects market-wide recession
indicators. No internal operational issues detected. Pattern:
macroeconomic pressure."

NOW ANALYZE THIS NEW DATASET:
Dataset: Q3 2025 sales data showing 15% decline in Region East

Provide analysis in the same style as the examples above.
─────────────────────────────────────────────────────────
```

### Why This Works
- The AI sees **how the expert thinks** through real examples
- Pattern recognition: the AI learns what to look for
- Style matching: the output follows the expert's documentation format
- No training required: the examples ARE the training, embedded in the prompt

---

### Generate Phase

The AI receives the augmented prompt and produces an analysis. This is where the actual output gets created.

### What Happens

```
Augmented Prompt → LLM (GPT-4, Claude, etc.) → Generated Analysis
```

### Continuing the Example

**AI receives**: The augmented prompt above (instructions + 2 examples + new data)

**AI generates**:
```
Generated Analysis:
───────────────────────────────────────────────────────
"The Q3 2025 decline of 15% in Region East shows characteristics
similar to the Q2 2024 case. Recommend investigating:
1. Regional supply chain status
2. Local market conditions specific to East region

The decline is isolated to one region (unlike the 2023 broad
decline), suggesting operational rather than macroeconomic factors.
Pattern: regional + likely operational issue."
───────────────────────────────────────────────────────
```

### What Makes Generation Effective

| Factor | Impact |
|--------|--------|
| Quality of retrieved examples | Better examples = better output |
| Number of examples | 3-5 examples usually optimal (too many = noise) |
| Similarity of examples | Closer matches = more relevant patterns |
| Clear expert annotations | AI mimics the style it sees |

### After Generation: The Human-in-the-Loop

```
AI generates analysis
        │
        ▼
Expert reviews output
        │
        ├─── Accept as-is → Store in database
        │
        ├─── Edit/correct → Store corrected version
        │
        └─── Reject → Expert writes from scratch
                            │
                            ▼
                      Store in database
                            │
                            ▼
            Database grows → Future retrievals improve
```

The cycle continues: each expert interaction makes the system smarter.

---

## LLM Calls via API: Possible Approaches

When building applications that use LLMs (like the Expert Learning System), you need a way to call the model programmatically. Here are the main approaches:

### Option 1: Paid API Services

Direct API access from major providers. Pay-per-use based on tokens (input/output text).

| Provider | Package | Model Examples | Notes |
|----------|---------|----------------|-------|
| **Anthropic** | `anthropic` | Claude Opus, Sonnet, Haiku | High quality, good for analysis |
| **OpenAI** | `openai` | GPT-4, GPT-4o | Well-documented, widely used |
| **Google** | `google-genai` | Gemini 1.5 Pro/Flash | Has free tier (15 req/min on Flash) |

**Setup pattern:**
```bash
pip install anthropic  # or openai, google-genai
# Get API key from provider's console
# Add billing/payment method (not needed for Google free tier)
# Use key in your code
```

**Cost:** Typically $1-20/month for learning and small projects. Scales with usage.

**Note:** A Claude Pro subscription ($20/month for claude.ai chat) is separate from API access. They require separate accounts and billing.

### Option 2: Local LLMs (Free)

Run open-source models on your own computer. No API key or ongoing costs.

**Tools:**
- **Ollama** - Simple CLI tool, easiest to set up
- **LM Studio** - GUI application, beginner-friendly
- **llama.cpp** - Lower-level, more control

**Popular local models:**
- Llama 3 (Meta) - Best overall quality
- Mistral - Good balance of speed/quality
- Phi-3 (Microsoft) - Smaller, faster

**Example with Ollama:**
```bash
# Install Ollama from ollama.com, then:
ollama pull llama3
ollama run llama3 "Analyze this data..."
```

**Hardware requirements:**
- Minimum: 8GB RAM
- Recommended: 16GB+ RAM
- Better with dedicated GPU (but not required)

**Tradeoffs:**
| Pros | Cons |
|------|------|
| Completely free | Requires decent hardware |
| Private (data stays local) | Quality below Claude/GPT-4 |
| No rate limits | Setup more complex |
| Works offline | Slower on CPU-only machines |

### Option 3: Hybrid/Manual Approach

Use a chat interface (like Claude Pro or Claude Code) manually instead of automating.

**Workflow:**
1. Prepare your data
2. Paste into chat interface
3. Ask for analysis
4. Copy results out

**Best for:**
- Learning and experimentation
- Low-volume analysis
- Prototyping before building automation

### Which Approach to Choose?

| Situation | Recommended Approach |
|-----------|---------------------|
| Learning concepts | Manual via Claude Code (no extra cost) |
| Want to code without paying | Local LLM (Ollama) |
| Building real application | Paid API (Anthropic or OpenAI) |
| Need free + decent quality | Google Gemini free tier |
| Data privacy critical | Local LLM |

### Switching Between Providers

The code structure stays similar across providers. Main changes:
- Import statement
- Client initialization
- Model name in API call

This makes it relatively easy to start with one provider and switch later.

---

## Realistic Limits of Current RAG Implementation

Understanding where this approach works well and where it struggles helps set appropriate expectations.

### What Works Well

| Data Characteristic | Why It Works |
|---------------------|--------------|
| **Tabular/structured data** | Clear fields, easy to embed and compare |
| **10-50 input fields** | Fits in context, embeddings capture meaning |
| **Single record per analysis** | Clear scope, focused patterns |
| **Explicit patterns** | "If glucose > 200, then high risk" - learnable |
| **Consistent expert logic** | AI can mimic reproducible reasoning |

### Where It Struggles

| Data Characteristic | Why It's Hard |
|---------------------|---------------|
| **Very large records** | Context window limits (~128K tokens for Gemini) |
| **Time-series / sequences** | Embeddings lose temporal relationships |
| **Images / mixed media** | Text embeddings don't capture visual data |
| **Multi-table relationships** | Joins and dependencies hard to represent in flat text |
| **Tacit expert knowledge** | "I just know from experience" can't be documented |
| **Rare edge cases** | Need many examples to cover unusual patterns |

### Practical Boundaries

**Record size:** Works well up to ~2-3 pages of text per record. Beyond that, important details get lost in the embedding.

**Field count:** 10-50 fields is comfortable. Hundreds of fields dilute the signal.

**Pattern complexity:**
- Single-factor patterns: Easy (elevated glucose = diabetes risk)
- Two-factor correlations: Good (BUN + creatinine = kidney)
- Three+ factor interactions: Harder, needs more examples
- Temporal patterns: Difficult (trending over time)

**Corpus size needed:**
- Simple domain: 50-100 examples
- Moderate complexity: 200-500 examples
- Complex with many edge cases: 500+ examples

### Patient Bloodwork Use Case Assessment

The current use case (patient bloodwork) is a **good fit** because:
- 10 numeric fields + 2 categorical fields
- Clear reference ranges
- Established clinical patterns
- Expert reasoning is mostly explicit
- Single-visit snapshot (no time component)

### When to Consider Other Approaches

| Situation | Better Alternative |
|-----------|-------------------|
| Need temporal analysis | Time-series models, LSTM |
| Image interpretation | Vision models, multimodal LLMs |
| Complex multi-table data | Graph databases, structured queries |
| Real-time decisions | Fine-tuned smaller models |
| Highly regulated (audit trail) | Rule-based systems with RAG augmentation |

### Key Takeaway

RAG with embeddings is excellent for **structured data with explicit patterns** where expert reasoning can be documented. It's not ideal for **temporal, visual, or deeply interconnected data** where relationships span beyond what text embeddings can capture.

---

## External Medical Data Sources

Reference information for integrating with electronic medical records and public biomedical databases.

---

### EMR Integration via API

Electronic Medical Record (EMR) systems can be accessed programmatically for data exchange.

#### Common Standards

**HL7 FHIR (Fast Healthcare Interoperability Resources)**
- Modern REST-based standard, uses JSON/XML
- Becoming the industry standard (mandated by US regulations since 2020)
- Resources like Patient, Observation, Condition, MedicationRequest

**HL7 v2**
- Older message-based standard (pipe-delimited format)
- Still widely used for lab results, ADT (admit/discharge/transfer)

#### Major EMR Systems & Their APIs

| EMR | API Approach |
|-----|--------------|
| **Epic** | Epic on FHIR, App Orchard marketplace |
| **Cerner** (Oracle Health) | FHIR R4, Millennium platform |
| **Allscripts** | FHIR, Open API |
| **athenahealth** | athenaOne API (proprietary + FHIR) |
| **Meditech** | FHIR (newer versions) |

#### Authentication

- **SMART on FHIR** - OAuth 2.0 framework specifically for healthcare
- Supports both patient-facing and provider-facing apps
- Scopes define what data the app can access

#### Key Challenges

1. **Access** - Most EMRs require vendor approval and contracts with healthcare organizations
2. **HIPAA compliance** - Must handle PHI securely
3. **Certification** - Apps may need ONC certification for clinical use
4. **Testing** - Sandbox environments vary in quality

#### Getting Started

- Epic: developer.epic.com (free sandbox)
- Cerner: developer.cerner.com
- SMART on FHIR sandbox: launch.smarthealthit.org

---

### Public Medical Databases

Free and open databases for medical/biomedical reference data.

#### OMIM (Online Mendelian Inheritance in Man)

- Database of human genes and genetic disorders
- **API access**: Requires free registration at omim.org/api
- Returns JSON/XML
- Rate limited (can request higher limits)
- Good for: gene-disease relationships, phenotypes, inheritance patterns

#### NCBI Databases (National Center for Biotechnology Information)

All accessible via **E-utilities API** (eutils.ncbi.nlm.nih.gov):

| Database | Content |
|----------|---------|
| **PubMed** | Biomedical literature |
| **ClinVar** | Clinical variant interpretations |
| **dbSNP** | Genetic variants |
| **Gene** | Gene records |
| **MedGen** | Medical genetics information |

- Free, no API key required (but recommended for higher rate limits)
- Returns XML or JSON

#### Other Key Databases

| Database | Content | Access |
|----------|---------|--------|
| **UniProt** | Protein sequences/function | REST API, free |
| **DrugBank** | Drug data | Free for academic, paid for commercial |
| **OpenFDA** | FDA drug/device adverse events | REST API, free |
| **RxNorm** | Drug nomenclature | REST API via NIH |
| **HPO** | Human Phenotype Ontology | Free download + API |
| **DisGeNET** | Gene-disease associations | Free academic |

#### Medical Terminologies

| Standard | Purpose | Access |
|----------|---------|--------|
| **SNOMED CT** | Clinical terms | Free via UMLS (requires account) |
| **ICD-10** | Diagnosis codes | Free via CMS/WHO |
| **LOINC** | Lab test codes | Free registration |
| **RxNorm** | Drug codes | Free via NIH |

#### Python Libraries

```python
# Biopython - NCBI access
from Bio import Entrez

# PyMedTermino - medical terminologies
# mygene - gene queries
```

---
