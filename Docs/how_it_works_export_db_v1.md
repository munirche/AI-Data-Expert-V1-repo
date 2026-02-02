# How Export DB Works - V1

A utility script to export LanceDB databases to CSV and JSON formats.

**Related files:**
- Script: `Code/export_db_v1.py`
- Databases: Any folder ending with `_db/`

---

## Purpose

LanceDB stores data in a binary format (`.lance` files) that can't be opened with Excel or text editors. This script exports the data to formats you can easily view:

| Format | Opens With |
|--------|------------|
| CSV | Excel, Google Sheets, any spreadsheet app |
| JSON | Any text editor, VS Code, Notepad++ |

---

## Quick Start

```bash
# See what databases exist
python Code/export_db_v1.py --list

# See database info (tables and record counts)
python Code/export_db_v1.py --info

# Export to CSV (opens in Excel)
python Code/export_db_v1.py --csv

# Export to JSON (opens in text editor)
python Code/export_db_v1.py --json

# Export to both formats
python Code/export_db_v1.py --csv --json
```

---

## Command Options

| Option | What it does |
|--------|--------------|
| `--list` | Show all available databases |
| `--info` | Show tables and record counts |
| `--csv` | Export to CSV format |
| `--json` | Export to JSON format |
| `--db PATH` | Specify which database to use |
| `--output-dir PATH` | Where to save exported files |

---

## Examples

### List Available Databases
```bash
python Code/export_db_v1.py --list
```
Output:
```
Available databases:
  expert_learning_system_v1_db
```

### View Database Info
```bash
python Code/export_db_v1.py --info
```
Output:
```
Database: expert_learning_system_v1_db
Tables: 1
----------------------------------------
  expert_annotations: 4 records
```

### Export to CSV
```bash
python Code/export_db_v1.py --csv
```
Creates: `expert_learning_system_v1.csv` (named after the database)

### Export to Specific Directory
```bash
python Code/export_db_v1.py --csv --json --output-dir exports
```
Creates:
- `exports/expert_learning_system_v1.csv`
- `exports/expert_learning_system_v1.json`

---

## What Gets Exported

All table columns except `vector` (the embedding data - just numbers, not useful to view).

| Column | Description |
|--------|-------------|
| annotation_id | Unique identifier |
| timestamp | When the annotation was created |
| expert_id | Who created it |
| dataset_summary | Brief description of the data |
| dataset_details | The actual raw data (tables, numbers, etc.) |
| expert_analysis | The expert's findings |
| patterns | JSON string of detected patterns |
| tags | Comma-separated category tags |
| text | The text that was embedded |

---

## Auto-Detection

If only one database exists, the script auto-detects it:
```bash
python Code/export_db_v1.py --csv
# Output: Auto-detected database: expert_learning_system_v1_db
```

If multiple databases exist, you must specify which one:
```bash
python Code/export_db_v1.py --db expert_learning_system_v1_db --csv
```

---

## Output Files

Exported files are named after the database (without the `_db` suffix):
- Database `expert_learning_system_v1_db` → `expert_learning_system_v1.csv` / `expert_learning_system_v1.json`

Files are saved to current directory by default, or use `--output-dir` to specify location.
