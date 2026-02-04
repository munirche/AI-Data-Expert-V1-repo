# How the Expert Learning System Works - V2

A practical guide to the CLI-based RAG system that learns from expert annotations.

**Related files:**
- Engine: `Code/expert_learning_system_v2.py`
- CLI: `Code/cli_v2.py`
- Configuration: `config.json`
- Corpus: `corpus/corpus.csv`
- Database: `expert_learning_system_v2_db/`

---

## What Changed from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Interface | Script with hardcoded examples | CLI with commands |
| Configuration | Embedded in code | External config.json |
| Test data | 3 hardcoded cases | 50-case corpus file |
| AI output | Free-form text | Structured JSON |
| Use case | Revenue analysis (fixed) | Any domain (configurable) |

---

## Core Concepts

### Separation of Engine and Use Case

The system is split into two parts:

**Engine (generic):** Handles RAG workflow, embeddings, database, AI calls. Has no domain-specific terms.

**Use Case (configurable):** Defines the domain - field names, risk levels, pattern types, prompts. Lives in config.json.

This means you can switch from "patient bloodwork analysis" to "revenue analysis" by changing the configuration file, not the code.

### The Three Data Stores

| Store | Format | Purpose | Changes? |
|-------|--------|---------|----------|
| **Corpus** | CSV | Ground truth test cases | Rarely (reference data) |
| **Database** | LanceDB | Where the system learns | Yes (grows over time) |
| **Config** | JSON | Use case definition | Rarely (domain setup) |

---

## The Learning Workflow

```
1. Load examples from corpus    --> Seeds the database with expert annotations
2. New record needs analysis    --> System retrieves similar past cases
3. AI generates structured output --> Summary, risk, patterns, actions
4. Expert reviews the output    --> Accept, Edit, or Reject
5. Approved analysis is stored  --> Database grows, AI improves
```

---

## What is an Annotation?

An annotation captures an expert's complete analysis of a record:

| Field | Purpose |
|-------|---------|
| record_id | Links back to source data |
| record_data | The original input (stored as JSON) |
| summary | Brief description (used for similarity search) |
| analysis | Detailed expert findings |
| risk_assessment | Risk level (low/moderate/high/critical) |
| patterns | Structured list of detected patterns |
| recommended_actions | What to do next |
| additional_tests | Further investigation needed |
| tags | Categories for filtering |

The **summary + analysis** text gets converted to an embedding vector for similarity search.

---

## How Similarity Search Works

1. **Text to numbers:** The query text is converted to a 768-dimensional vector using Gemini's embedding model
2. **Vector comparison:** LanceDB compares this vector against all stored annotation vectors
3. **Distance calculation:** Closer vectors = more similar content
4. **Top matches returned:** The 3 most similar past annotations are retrieved

This allows the system to find "semantically similar" cases - not just keyword matches, but cases with similar meaning.

---

## The AI Analysis Pipeline

When you run an analysis, here's what happens:

### Step 1: Load the Record
The record data is loaded from the corpus or a JSON file. Data fields (age, glucose, etc.) are extracted.

### Step 2: Retrieve Similar Cases
The system builds a query from the record fields and searches the database for similar past annotations.

### Step 3: Build the Prompt
A prompt is constructed using:
- The similar examples (showing how experts analyzed similar cases)
- The new record data
- Instructions to return structured JSON

### Step 4: Call the LLM
The prompt is sent to Gemini, which generates a structured JSON response with all annotation fields.

### Step 5: Parse and Display
The JSON response is parsed into individual fields (summary, risk, patterns, etc.) and displayed for expert review.

### Step 6: Expert Decision
- **Accept:** Save the AI output as-is
- **Edit:** Modify the AI output in a JSON file, then save
- **Reject:** Write your own analysis from scratch
- **Skip:** Don't save anything

---

## CLI Commands Overview

| Command | Purpose |
|---------|---------|
| `load` | Load records from corpus into database |
| `list` | Show annotations in database |
| `stats` | Database statistics and corpus progress |
| `search` | Find similar past annotations |
| `analyze` | AI analyzes a record, expert reviews |
| `add` | Add annotation from JSON file |
| `delete` | Remove annotation(s) from database |
| `reset` | Clear the database |

### Educational Mode

Add `--explain` to `load` or `analyze` to see step-by-step what the system is doing:
- Which files are being read
- What text is being embedded
- The full prompt sent to the LLM
- How the response is parsed

---

## The Current Use Case: Patient Bloodwork

For development, the system is configured for clinical lab analysis:

**Input fields:** age, sex, medical_history, glucose, bun, creatinine, sodium, potassium, hemoglobin, wbc

**Output:** Risk assessment, clinical analysis, detected patterns, recommended actions, additional tests

**Pattern types:** abnormal_value, correlation, risk_indicator, trend, normal

**Risk levels:** low, moderate, high, critical

The corpus contains 50 simulated cases covering normal results, single abnormalities, kidney patterns, diabetes patterns, infections, electrolyte issues, and complex multi-factor cases.

---

## How It Improves Over Time

| Database Size | AI Capability | Expert Role |
|---------------|---------------|-------------|
| 0-10 annotations | Limited context, generic output | Does most analysis manually |
| 10-30 annotations | Better pattern matching | Reviews AI suggestions, corrects often |
| 30-50+ annotations | Good domain knowledge | Handles routine cases, reviews edge cases |

The key insight: **every accepted or corrected analysis becomes a new training example.** The more the expert uses the system, the smarter it gets.

---

## Typical Workflows

### Testing the AI

1. Reset database and load 10 training cases
2. Analyze case 11 with `--compare` to see ground truth
3. Check if AI matches expert assessment
4. Load more cases, repeat

### Production Use

1. Load initial corpus into database
2. Analyze new incoming records
3. Expert reviews, edits as needed
4. Approved analyses grow the database
5. AI quality improves over time

---

## Key Terms

| Term | Definition |
|------|------------|
| **Corpus** | CSV file with ground truth test cases |
| **Embedding** | Vector representation of text meaning (768 numbers) |
| **RAG** | Retrieval-Augmented Generation - find examples before generating |
| **Similarity Score** | How close a past case is to the current one (0-100%) |
| **Structured Output** | AI returns JSON with specific fields, not free text |

---

## Files Summary

| File | Purpose |
|------|---------|
| `Code/expert_learning_system_v2.py` | Generic RAG engine |
| `Code/cli_v2.py` | Command-line interface |
| `config.json` | Use case configuration |
| `corpus/corpus.csv` | 50 ground truth test cases |
| `expert_learning_system_v2_db/` | LanceDB vector database |
| `Docs/Plan_for_expert_learning_system_v2.md` | Implementation plan and progress |
