# How Launch Lance Viewer Works

A utility script to open the Lance Data Viewer in your browser for inspecting LanceDB tables.

**Related files:**
- Script: `Code/launch_lance_viewer.py`
- Databases: `expert_learning_system_v1_db/`, `expert_learning_system_v2_db/`

---

## Purpose

LanceDB stores data in binary `.lance` files that you can't open directly. While `export_db_v1.py` exports to CSV/JSON, sometimes you want to browse the data interactively without exporting.

Lance Data Viewer is a web-based tool that lets you:
- Browse tables visually
- See all records and columns
- Inspect data without exporting

---

## Quick Start

```powershell
python Code/launch_lance_viewer.py           # View all databases
python Code/launch_lance_viewer.py v2        # View V2 database only
python Code/launch_lance_viewer.py v1        # View V1 database only
python Code/launch_lance_viewer.py --list    # List available databases
```

Then open http://localhost:8080 in your browser.

Press `Ctrl+C` in PowerShell to stop.

---

## Prerequisites

- **Docker Desktop** must be installed and running
- First run will download the Docker image (~200MB)

---

## What the Script Does

1. **Stops any existing viewer** containers (clears port 8080)
2. **Finds all .lance tables** in the configured databases
3. **Mounts tables** with unique names (e.g., `v1_expert_annotations.lance`)
4. **Runs the container** which starts a web server on port 8080
5. **Waits for Ctrl+C** to stop the container

---

## Command Options

| Command | Purpose |
|---------|---------|
| `python Code/launch_lance_viewer.py` | View all databases (default) |
| `python Code/launch_lance_viewer.py all` | View all databases |
| `python Code/launch_lance_viewer.py v1` | View V1 database only |
| `python Code/launch_lance_viewer.py v2` | View V2 database only |
| `python Code/launch_lance_viewer.py --list` | List available databases and tables |

---

## Configuration

The script has these settings at the top:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PROJECT_PATH` | `AI Data Expert V1` | Project folder path |
| `DATABASES` | `v1`, `v2` | Available database mappings |
| `IMAGE` | `lancedb-0.24.3` | Docker image version |
| `PORT` | `8080` | Local port for the web interface |
| `CONTAINER_NAME` | `lance-viewer` | Docker container name |

---

## How Multiple Databases Work

When viewing all databases, the script:
1. Finds all `.lance` folders in each database
2. Mounts each with a unique name: `v1_tablename.lance`, `v2_tablename.lance`
3. All tables appear in the viewer's table list

Example table names in viewer:
- `v1_expert_annotations.lance` (from V1)
- `v2_expert_annotations.lance` (from V2)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Docker not found" | Start Docker Desktop first |
| Port 8080 in use | Script auto-clears the port; if it fails, run `docker ps` and stop manually |
| No tables shown | Check that database folders exist and contain `.lance` files |
| Container won't start | Run `docker pull ghcr.io/gordonmurray/lance-data-viewer:lancedb-0.24.3` manually |

---

## When to Use This vs Export

| Use Case | Tool |
|----------|------|
| Quick browse of data | `launch_lance_viewer.py` |
| Share data with others | `export_db_v1.py --csv` |
| Open in Excel/Sheets | `export_db_v1.py --csv` |
| Process data in another script | `export_db_v1.py --json` |
