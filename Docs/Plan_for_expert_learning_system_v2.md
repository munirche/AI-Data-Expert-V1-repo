# Plan for Expert Learning System V2

This document captures the design decisions and implementation plan for V2 of the expert learning system, based on discussions with Claude Opus 4.5 on 02/02/2026.

---

## Overview

**V1 Status:** Demo script showing RAG workflow. Kept as reference, not for production use.

**V2 Goal:** A functional, generic CLI tool that:
- Learns from expert annotations
- Works with any domain/use case
- Provides commands to add, analyze, search, and manage annotations
- Separates the engine from use-case specifics

---

## Core Design Principle: Separation of Concerns

The engine and use case are kept strictly separate.

| Component | Contains | Example |
|-----------|----------|---------|
| **Engine** (V2 code) | Generic RAG workflow, CLI, database operations | `record`, `annotation`, `analysis` |
| **Use Case** (config + data) | Domain-specific schema, prompts, vocabulary | `patient_id`, `glucose`, `risk_assessment` |

The engine code has **no domain-specific terms**. All domain specifics come from configuration files.

---

## Project Structure

```
AI Data Expert V1/
├── Code/
│   ├── expert_learning_system_v1.py      # V1 (reference)
│   ├── expert_learning_system_v2.py      # V2 engine
│   ├── cli_v2.py                         # V2 CLI interface
│   ├── export_db_v1.py
│   └── launch_lance_viewer.py
├── Docs/
│   ├── Plan_for_expert_learning_system_v2.md   # This document
│   └── ... (existing docs)
├── expert_learning_system_v1_db/         # V1 database
├── expert_learning_system_v2_db/         # V2 database
├── corpus/
│   └── corpus.csv                        # Ground truth test cases
├── config.json                           # Use case configuration
├── memory.md
└── requirements.txt
```

---

## Data Stores

| Store | Format | Purpose | Mutable |
|-------|--------|---------|---------|
| **Corpus** | CSV | Ground truth, test cases, reference | No (read-only) |
| **Database** | LanceDB | Where system learns, stores annotations | Yes |
| **Config** | JSON | Use case definition | Rarely |

---

## Configuration (config.json)

Defines the use case without changing engine code.

```json
{
  "use_case_name": "Display name for this use case",
  "database_path": "./expert_learning_system_v2_db",
  "corpus_path": "./corpus/corpus.csv",

  "record_id_field": "record_id",
  "data_fields": ["field1", "field2", "field3"],

  "annotation_fields": {
    "summary": "required",
    "analysis": "required",
    "risk_assessment": "optional",
    "patterns": "optional",
    "recommended_actions": "optional",
    "additional_tests": "optional",
    "tags": "optional"
  },

  "pattern_vocabulary": ["abnormal_value", "correlation", "risk_indicator", "normal"],
  "risk_levels": ["low", "moderate", "high", "critical"],

  "ai_prompt_template": "You are an expert analyst. Based on these examples:\n{examples}\n\nAnalyze this new record:\n{record}"
}
```

---

## Corpus Structure (corpus.csv)

Single CSV file containing all test cases with ground truth annotations.

**Columns:**
- Record data fields (defined in config)
- Annotation fields (summary, analysis, risk_assessment, etc.)

The corpus serves as:
1. Ground truth for testing
2. Source for seeding the database
3. Evaluation set for measuring AI performance

---

## Database Schema

Stored in LanceDB (expert_learning_system_v2_db/).

| Field | Type | Purpose |
|-------|------|---------|
| `annotation_id` | string | Unique identifier |
| `record_id` | string | Reference to source record |
| `timestamp` | string | When annotation was created |
| `record_data` | string | JSON of the original record |
| `summary` | string | Brief description (for search) |
| `analysis` | string | Full expert analysis |
| `risk_assessment` | string | Risk level |
| `patterns` | string | JSON array of pattern objects |
| `recommended_actions` | string | JSON array of actions |
| `additional_tests` | string | JSON array of tests |
| `tags` | string | Comma-separated tags |
| `text` | string | Field to embed (summary + analysis) |
| `vector` | Vector | Embedding (auto-generated) |

---

## CLI Commands

### load

Load records from corpus into database.

```
cli_v2.py load N                  # Load first N records
cli_v2.py load --record 5         # Load specific record
cli_v2.py load --record 5-10      # Load range
```

### add

Add annotation manually from file.

```
cli_v2.py add --file annotation.json
```

**annotation.json structure:**
```json
{
  "record_id": "051",
  "record_data": { ... },
  "summary": "Brief description",
  "analysis": "Detailed analysis...",
  "risk_assessment": "moderate",
  "patterns": [...],
  "recommended_actions": [...],
  "tags": [...]
}
```

**Validation (Phase 1):** Required fields present, correct types.
**Validation (Phase 2):** Enum values match config vocabulary.

### analyze

AI analyzes a record, expert reviews, result saved.

```
cli_v2.py analyze --record 15           # From corpus
cli_v2.py analyze --file record.json    # From file
cli_v2.py analyze --record 15 --compare # Show ground truth comparison
cli_v2.py analyze --record 11-20 --batch # Batch with metrics only
```

**Interactive workflow:**
1. Load record
2. Retrieve similar past annotations
3. AI generates structured JSON analysis
4. Display each field: summary, risk, analysis, patterns, actions, tests
5. Expert chooses: Accept / Edit / Reject / Skip
6. Save all structured fields to database

**AI output format:** JSON with summary, analysis, risk_assessment, patterns (array), recommended_actions (array), additional_tests (array).

**Edit approach:** Save to temp JSON file with all fields, user edits externally, press Enter to continue.

**Compare mode:** Shows ground truth from corpus alongside AI output.

**Batch mode:** No prompts, no saving, just metrics comparison.

### list

Show annotations in database.

```
cli_v2.py list                    # Summary view
cli_v2.py list --full             # Full analysis text
cli_v2.py list --limit 10         # Limit results
cli_v2.py list --tag X            # Filter by tag
```

### search

Find similar past annotations.

```
cli_v2.py search "query text"
cli_v2.py search "query" --limit 10
cli_v2.py search "query" --full
```

### stats

Database statistics.

```
cli_v2.py stats
```

**Output:**
- Total annotations
- Date range
- Risk distribution
- Top tags
- Corpus progress (loaded vs remaining)

### reset

Clear database.

```
cli_v2.py reset --confirm              # Full reset
cli_v2.py reset --reload 10 --confirm  # Reset and load 10 from corpus
```

Requires `--confirm` flag. Prompts for typed confirmation.

### delete

Delete specific annotation(s) from database.

```
cli_v2.py delete --id <annotation_id>     # Delete by annotation ID
cli_v2.py delete --record <record_id>     # Delete all annotations for a record
```

Shows what will be deleted and prompts for confirmation.

---

## Workflow Examples

### Testing Workflow

```
# Start fresh
cli_v2.py reset --confirm

# Load initial training set
cli_v2.py load 10

# Test AI on next case
cli_v2.py analyze --record 11 --compare

# Add more training data
cli_v2.py load --record 11

# Test again
cli_v2.py analyze --record 12 --compare

# Check progress
cli_v2.py stats
```

### Batch Evaluation

```
# Load training set
cli_v2.py reset --reload 10 --confirm

# Evaluate on test set
cli_v2.py analyze --record 11-30 --batch

# Results show accuracy metrics
```

### Production Workflow (Future)

```
# Analyze new incoming data
cli_v2.py analyze --file new_record.json

# Expert reviews, edits, saves
# Database grows, system improves
```

---

## Initial Use Case: Patient Bloodwork

For development and testing, the first use case is patient bloodwork analysis.

**Domain specifics (in config.json, not in engine code):**
- Records: Single patient visit with lab results
- Data fields: age, sex, medical_history, glucose, bun, creatinine, sodium, potassium, hemoglobin, wbc
- Expert output: Risk assessment, explanation, next steps, additional tests
- Patterns: abnormal_value, correlation, risk_indicator, normal

**Corpus:** ~50 simulated cases with expert annotations
- Normal: 10 cases
- Single abnormal: 12 cases
- Kidney indicators: 6 cases
- Diabetes patterns: 6 cases
- Infection signs: 4 cases
- Electrolyte imbalance: 4 cases
- Multi-factor risk: 4 cases
- Age/sex adjusted: 4 cases

---

## Implementation Phases

### Phase 1: Core CLI (Completed 02/02/2026)
- [x] Set up project structure
- [x] Create config.json for use case
- [x] Implement engine with generic schema
- [x] Implement commands: load, list, stats, reset
- [x] Create initial corpus (15 cases)
- [x] Test basic workflow

### Phase 2: AI Analysis (In Progress)
- [x] Implement analyze command
- [x] Implement search command
- [x] Add compare mode
- [x] Add batch mode
- [x] Add delete command
- [x] Structured AI output (JSON with all fields: summary, analysis, risk, patterns, actions, tests)
- [ ] Expand corpus to 50 cases
- [ ] Measure baseline accuracy

### Phase 3: Refinement
- [ ] Add validation against config vocabulary
- [ ] Improve prompts based on results
- [x] Add add command (manual entry)
- [ ] Performance tuning

### Phase 4: Production Readiness
- [ ] Error handling
- [ ] Logging
- [x] Documentation (CLI, lance viewer)
- [ ] Real data testing

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Pattern detection rate | AI finds same patterns as expert |
| Risk assessment accuracy | AI matches expert risk level |
| Recommendation overlap | AI suggests similar actions |
| Improvement curve | Accuracy increases as DB grows |

---

## Dependencies

From requirements.txt:
- google-genai (LLM + embeddings)
- lancedb (vector database)
- pandas (CSV handling)

---

## References

- V1 script: `Code/expert_learning_system_v1.py`
- V1 documentation: `Docs/how_it_works_expert_learning_system_v1.md`
- V1 discussion: `Docs/Discussion_expert_learning_system_v1.md`
- Concepts: `Docs/concepts.md`
