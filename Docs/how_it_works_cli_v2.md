# How CLI V2 Works

Command-line interface for the Expert Learning System V2.

**Related files:**
- CLI: `Code/cli_v2.py`
- Engine: `Code/expert_learning_system_v2.py`
- Config: `config.json`
- Corpus: `corpus/corpus.csv`
- Database: `expert_learning_system_v2_db/`

---

## Overview

The CLI provides commands to manage expert annotations and test the RAG learning system. It reads configuration from `config.json` and stores annotations in a LanceDB database.

---

## Quick Start

```powershell
python Code/cli_v2.py stats              # Check database status
python Code/cli_v2.py load 5             # Load 5 records from corpus
python Code/cli_v2.py list               # List all annotations
python Code/cli_v2.py search "query"     # Find similar annotations
python Code/cli_v2.py reset --confirm    # Clear database
```

---

## Commands

### stats

Show database statistics.

```powershell
python Code/cli_v2.py stats
```

**Output includes:**
- Total annotations in database
- Corpus size and progress (loaded vs remaining)
- Risk level distribution
- Top tags

**Example output:**
```
=== Database Statistics ===

Annotations: 5

Corpus: 15 cases
  Loaded: 5 (33%)
  Remaining: 10

Risk distribution:
  low: 2 (40%)
  moderate: 2 (40%)
  high: 1 (20%)

Top tags:
  routine: 2
  kidney: 2
```

---

### load

Load records from corpus into database.

```powershell
python Code/cli_v2.py load 5             # Load first 5 records
python Code/cli_v2.py load --record 10   # Load specific record by ID
python Code/cli_v2.py load --record 5-10 # Load range of records
```

| Option | Purpose |
|--------|---------|
| `N` | Load first N records |
| `--record ID` | Load specific record by ID |
| `--record N-M` | Load records from row N to M |

**Note:** Record IDs in corpus may not have leading zeros (use `10` not `010`).

---

### list

List annotations in database.

```powershell
python Code/cli_v2.py list               # Summary view
python Code/cli_v2.py list --full        # Show full analysis text
python Code/cli_v2.py list --limit 10    # Limit results
python Code/cli_v2.py list --tag kidney  # Filter by tag
```

| Option | Purpose |
|--------|---------|
| `--full` | Show complete analysis for each annotation |
| `--limit N` | Show only N annotations |
| `--tag X` | Filter by tag |

**Example output:**
```
ID         Record   Risk       Summary
--------------------------------------------------------------------------------
74da3d5c   1        low        45F routine checkup, all values normal
4f335799   2        moderate   58M elevated glucose with cardiac risk factors

Total: 2 annotations
```

---

### search

Find similar annotations using semantic search.

```powershell
python Code/cli_v2.py search "diabetic patient with kidney issues"
python Code/cli_v2.py search "elevated glucose" --limit 3
python Code/cli_v2.py search "anemia" --full
```

| Option | Purpose |
|--------|---------|
| `--limit N` | Return top N results (default: 5) |
| `--full` | Show full analysis text |

**Example output:**
```
Similar annotations for: "diabetic patient with kidney issues"

#1 (56% match): 4f335799
   58M elevated glucose with cardiac risk factors

#2 (55% match): cfa38d94
   72M diabetic with kidney function concerns
```

The match percentage indicates semantic similarity to your query.

---

### reset

Clear all annotations from database.

```powershell
python Code/cli_v2.py reset --confirm              # Clear database
python Code/cli_v2.py reset --reload 5 --confirm   # Clear and reload 5 records
```

| Option | Purpose |
|--------|---------|
| `--confirm` | Required flag to execute reset |
| `--reload N` | After clearing, load N records from corpus |

**Safety:** Requires `--confirm` flag and typed confirmation when annotations exist.

---

### add

Add annotation manually from JSON file.

```powershell
python Code/cli_v2.py add --file annotation.json
```

**annotation.json structure:**
```json
{
  "record_id": "051",
  "record_data": {
    "age": 55,
    "sex": "M",
    "glucose": 105
  },
  "summary": "55M with borderline glucose",
  "analysis": "Glucose slightly elevated at 105...",
  "risk_assessment": "low",
  "patterns": [{"type": "abnormal_value", "location": "glucose"}],
  "recommended_actions": ["Dietary counseling"],
  "additional_tests": ["HbA1c"],
  "tags": ["glucose", "borderline"]
}
```

**Required fields:** `record_id`, `summary`, `analysis`

---

### analyze

Analyze a record using AI (Phase 2 - coming soon).

```powershell
python Code/cli_v2.py analyze --record 15           # From corpus
python Code/cli_v2.py analyze --file record.json    # From file
python Code/cli_v2.py analyze --record 15 --compare # Compare with ground truth
python Code/cli_v2.py analyze --record 11-20 --batch # Batch mode
```

| Option | Purpose |
|--------|---------|
| `--record ID` | Analyze record from corpus |
| `--file PATH` | Analyze record from JSON file |
| `--compare` | Show ground truth comparison |
| `--batch` | Batch mode (no prompts, metrics only) |

---

## Workflow Examples

### Testing workflow

```powershell
# Start fresh
python Code/cli_v2.py reset --confirm

# Load training set
python Code/cli_v2.py load 10

# Check status
python Code/cli_v2.py stats

# Search for patterns
python Code/cli_v2.py search "kidney dysfunction"

# View specific annotations
python Code/cli_v2.py list --tag kidney --full
```

### Reset and reload

```powershell
# Clear and start with 5 records
python Code/cli_v2.py reset --reload 5 --confirm

# Verify
python Code/cli_v2.py stats
```

---

## Configuration

The CLI reads from `config.json` in the project root:

| Setting | Purpose |
|---------|---------|
| `database_path` | Where to store LanceDB data |
| `corpus_path` | Path to corpus CSV file |
| `record_id_field` | Column name for record IDs |
| `data_fields` | List of data columns |
| `ai_prompt_template` | Template for AI generation |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | Run `pip install -r requirements.txt` |
| Record not found | Check record ID format (no leading zeros) |
| Reset fails | Stop Lance Data Viewer first (releases file locks) |
| No results in search | Load some records first with `load` command |

---

## Data Flow

```
corpus.csv (ground truth)
    │
    ├── load ──────────────> expert_learning_system_v2_db/
    │                              │
    │                              ├── list (view)
    │                              ├── search (query)
    │                              ├── stats (metrics)
    │                              └── reset (clear)
    │
    └── analyze ──> AI generates ──> expert reviews ──> save to db
```
