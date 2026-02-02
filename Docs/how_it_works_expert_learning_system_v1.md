# How the Expert Learning System Works - V1

A simple explanation of the system and how it learns from expert annotations.

**Related files:**
- Script: `Code/expert_learning_system_v1.py`
- Database: `expert_learning_system_v1_db/`

---

## The Core Idea

The system teaches an AI to analyze data the way an expert would - by showing it examples.

**Without this system:** AI guesses based on general knowledge (might miss domain-specific patterns)

**With this system:** AI learns from real expert examples and mimics their approach

---

## The Learning Loop

```
1. Expert analyzes data       -->  Writes findings and tags patterns
2. System stores the example  -->  Converts to embeddings for search
3. New data arrives           -->  System finds similar past examples
4. AI generates analysis      -->  Mimics the expert's style
5. Expert reviews/corrects    -->  Corrections get stored
6. Repeat                     -->  AI improves over time
```

---

## What is an Annotation?

An annotation is an expert's analysis of a dataset, containing:

| Component | What it is |
|-----------|------------|
| Dataset Summary | Brief description of the data |
| Dataset Details | The actual raw data (tables, numbers, etc.) |
| Expert Analysis | The expert's written findings and recommendations |
| Patterns Found | Specific patterns detected (anomalies, trends, etc.) |
| Tags | Categories for easy filtering (e.g., "revenue", "anomaly") |

---

## Concrete Example

### The Data (Input)

```
Q1-Q4 2023 revenue data by region

Observations:
- Q3 had a 15% revenue drop across all regions
- West region performs 22% above average
- Q2 and Q4 appear to be peak quarters
```

### The Expert Annotation (Output)

**Dataset Summary:**
```
Q1-Q4 2023 revenue data by region
```

**Dataset Details (Raw Data):**
```
Quarter | East   | West   | North  | South  | Total
--------|--------|--------|--------|--------|--------
Q1      | $2.0M  | $3.2M  | $1.8M  | $1.5M  | $8.5M
Q2      | $2.3M  | $3.6M  | $2.0M  | $1.7M  | $9.6M
Q3      | $1.7M  | $2.7M  | $1.5M  | $1.3M  | $7.2M
Q4      | $2.1M  | $3.5M  | $2.0M  | $1.8M  | $9.4M
```

**Expert Analysis:**
```
Key Findings:
1. Significant anomaly in Q3 - revenue dropped 15% across all regions
2. Investigation revealed system outage during migration
3. Q4 shows recovery with 8% growth
4. West region consistently outperforms (22% above average)
5. Seasonality pattern: Q2 and Q4 are peak quarters

Recommendations:
- Exclude Q3 from baseline calculations
- Focus growth investments in West region strategy
- Prepare for seasonal spikes in Q2/Q4
```

**Patterns Found:**
```json
[
  {"type": "anomaly", "location": "Q3", "severity": "high"},
  {"type": "trend", "location": "West_region", "direction": "positive"},
  {"type": "seasonality", "period": "quarterly", "peaks": ["Q2", "Q4"]}
]
```

**Tags:**
```
revenue, anomaly, seasonality
```

---

## What Happens Behind the Scenes

### Step 1: Storing the Annotation

When an expert saves their annotation:

1. The analysis text gets converted to **embeddings** (a list of ~768 numbers)
2. These numbers capture the "meaning" of the text
3. Similar analyses will have similar numbers
4. Everything gets stored in the vector database (LanceDB)

### Step 2: Finding Similar Examples

When new data arrives:

1. The new data summary gets converted to embeddings
2. System searches for annotations with similar embeddings
3. Returns the top 3-5 most similar past analyses

### Step 3: AI Generates Analysis

The AI receives:
- The similar past examples (showing how experts analyzed similar data)
- The new data to analyze
- Instructions to follow the expert's style

The AI then produces an analysis mimicking the expert's approach.

### Step 4: Expert Reviews

The expert can:
- **Accept** - Analysis is good, store it
- **Edit** - Fix mistakes, store the corrected version
- **Reject** - Write from scratch

Any corrections become new training examples, making the system smarter.

---

## How It Improves Over Time

| Phase | Expert Work | AI Capability |
|-------|-------------|---------------|
| Early (0-50 examples) | Does everything manually | Limited, often wrong |
| Middle (50-200 examples) | Reviews AI suggestions, corrects often | Decent, needs guidance |
| Late (200+ examples) | Reviews only edge cases | Handles most routine analyses |

---

## Key Terms

| Term | Simple Definition |
|------|-------------------|
| **Annotation** | Expert's documented analysis of a dataset |
| **Embedding** | Numbers representing the meaning of text |
| **Vector Database** | Storage optimized for finding similar embeddings |
| **RAG** | Retrieval-Augmented Generation - finding examples before generating |
| **Similarity Score** | How close a past example is to the new data (0-100%) |
| **Confidence** | How sure the AI is based on available examples |

---

## Files in This Project

| File | Purpose |
|------|---------|
| `Code/expert_learning_system_v1.py` | Main system code |
| `expert_learning_system_v1_db/` | Vector database storage (local, not in git) |
| `Docs/concepts.md` | Technical concepts reference |
| `Docs/how_it_works_expert_learning_system_v1.md` | This file - simple explanation |

---

## What Happens When You Run the Script

When you run `python Code/expert_learning_system_v1.py`, here's what happens:

### Initialization
1. **System starts** - Creates connection to LanceDB (stored in `./expert_learning_system_v1_db` folder)
2. **Gemini client connects** - Uses your `GEMINI_API_KEY` environment variable

### Phase 1: Store Expert Annotations
The code stores 3 example annotations:

| # | Dataset | What Expert Found |
|---|---------|-------------------|
| 1 | Q1-Q4 2023 revenue by region | Q3 anomaly, West region strong, seasonal pattern |
| 2 | Customer churn Jan-Dec 2023 | Churn spike from competitor, high-value customers retained |
| 3 | Website traffic 2023 | Mobile up, desktop down, July bounce rate spike |

Each annotation:
- Gets a unique ID
- Text gets converted to **embeddings** (numbers representing meaning)
- Stored in LanceDB vector database

### Phase 2: AI Analyzes New Data
1. **New dataset arrives**: Q1 2024 revenue data
2. **Retrieval**: System searches LanceDB for similar past analyses (finds the Q1-Q4 2023 revenue annotation)
3. **Prompt built**: Combines retrieved examples + new data + instructions
4. **Gemini API called**: Generates analysis mimicking expert style
5. **Confidence calculated**: Based on how similar the retrieved examples are

### Phase 3: Expert Feedback
1. Expert reviews AI output and provides corrections
2. Corrected version gets stored as a new annotation
3. Database now has 4 annotations (system got smarter)

### Key Point
**Each run adds to the database.** If you run it 3 times, you'll have 12 annotations (3 initial + 1 feedback) x 3 runs. The database folder persists between runs.
