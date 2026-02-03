# How Launch Lance Viewer Works

A utility script to open the Lance Data Viewer in your browser for inspecting LanceDB tables.

**Related files:**
- Script: `Code/launch_lance_viewer.py`
- Database: `expert_learning_system_v1_db/`

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
python Code/launch_lance_viewer.py
```

Then open http://localhost:8080 in your browser.

Press `Ctrl+C` in PowerShell to stop.

---

## Prerequisites

- **Docker Desktop** must be installed and running
- First run will download the Docker image (~200MB)

---

## What the Script Does

1. **Builds the Docker command** with the correct paths and settings
2. **Runs the container** which starts a web server on port 8080
3. **Mounts your database folder** as read-only (`:ro`) so the viewer can read but not modify your data
4. **Waits for Ctrl+C** to stop the container

---

## Configuration

The script has these settings at the top:

| Variable | Value | Purpose |
|----------|-------|---------|
| `DB_PATH` | `expert_learning_system_v1_db` | Which database to view |
| `IMAGE` | `lancedb-0.24.3` | Docker image version |
| `PORT` | `8080` | Local port for the web interface |

To view a different database, edit `DB_PATH` in the script.

---

## The Docker Command

The script runs this command:

```powershell
docker run --rm -p 8080:8080 -v "C:/Users/munir/Projects/AI Data Expert V1/expert_learning_system_v1_db:/data:ro" ghcr.io/gordonmurray/lance-data-viewer:lancedb-0.24.3
```

| Flag | Meaning |
|------|---------|
| `--rm` | Remove container when stopped (clean up) |
| `-p 8080:8080` | Map port 8080 on your PC to the container |
| `-v ...:/data:ro` | Mount database folder as `/data` inside container, read-only |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Docker not found" | Start Docker Desktop first |
| Port 8080 in use | Stop other services using that port, or edit `PORT` in script |
| No tables shown | Check that the database path exists and contains `.lance` files |
| Container won't start | Run `docker pull ghcr.io/gordonmurray/lance-data-viewer:lancedb-0.24.3` manually |

---

## When to Use This vs Export

| Use Case | Tool |
|----------|------|
| Quick browse of data | `launch_lance_viewer.py` |
| Share data with others | `export_db_v1.py --csv` |
| Open in Excel/Sheets | `export_db_v1.py --csv` |
| Process data in another script | `export_db_v1.py --json` |
