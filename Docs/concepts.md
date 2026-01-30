# Concepts

A reference for key concepts encountered during research and development.

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

## Embeddings

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

---

## Augment Phase

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

## Generate Phase

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
